// OAE Compute Relay A2A Gateway — Supabase Edge Function v0.1
//
// Purpose:
// - Provide a lightweight HTTP gateway for agent-to-agent compute.
// - Accept quote requests.
// - Accept Stripe/payment event JSON and convert it into queue records.
// - Return queue records to a caller or future queue sink.
//
// Safety:
// - This function does not execute compute directly.
// - Webhook/write routes require custom auth if OAE_RELAY_SHARED_SECRET is set.
// - Stripe signature verification is TODO until STRIPE_WEBHOOK_SECRET is configured.

import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const PRODUCT = {
  name: "OAE Compute Relay Credit",
  priceCad: 5,
  stripeProductId: "prod_Ub2m82KsbFXy3T",
  stripePriceId: "price_1TbqgzFRuWOMBXuRYwxGaDQW",
  stripePaymentLinkId: "plink_1TbqhQFRuWOMBXuRuZgJzpZW",
  stripePaymentLink: "https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f",
};

const ROUTES: Record<string, { route: string; estimatedCostCad: number; notes: string }> = {
  text: { route: "template-local", estimatedCostCad: 0.001, notes: "Deterministic text artifact." },
  code: { route: "template-local-code", estimatedCostCad: 0.001, notes: "Deterministic code skeleton." },
  "hf-search": { route: "future-hf-search", estimatedCostCad: 0.02, notes: "Future HF model/dataset/Space search." },
  llm: { route: "future-hf-inference", estimatedCostCad: 0.10, notes: "Future HF inference route with cost cap." },
  gpu: { route: "future-paid-gpu", estimatedCostCad: 2.00, notes: "Requires deposit and explicit margin check." },
};

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET,POST,OPTIONS",
      "access-control-allow-headers": "content-type,authorization,x-oae-relay-secret,stripe-signature",
    },
  });
}

async function sha256(input: unknown): Promise<string> {
  const raw = typeof input === "string" ? input : JSON.stringify(input);
  const data = new TextEncoder().encode(raw);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function requireSharedSecret(req: Request): Response | null {
  const expected = Deno.env.get("OAE_RELAY_SHARED_SECRET") || "";
  if (!expected) return null;
  const actual = req.headers.get("x-oae-relay-secret") || "";
  if (actual !== expected) {
    return json({ status: "blocked", reason: "invalid_or_missing_oae_relay_secret" }, 401);
  }
  return null;
}

function quoteJob(job: any) {
  const taskType = String(job?.task_type || "text").toLowerCase();
  const route = ROUTES[taskType] || ROUTES.text;
  const credit = Number(job?.credit_cad ?? PRODUCT.priceCad ?? 0);
  const maxCost = Number(job?.max_compute_cost_cad ?? route.estimatedCostCad ?? 0);
  const privacy = String(job?.privacy_level || "unknown");
  const estimatedCost = route.estimatedCostCad;
  const riskFlags: string[] = [];

  if (!job?.paid) riskFlags.push("UNPAID_JOB");
  if (estimatedCost > credit) riskFlags.push("NEGATIVE_MARGIN");
  if (!["non_sensitive", "public", "unknown"].includes(privacy)) riskFlags.push("PRIVACY_REVIEW_REQUIRED");
  if (taskType === "gpu") riskFlags.push("GPU_REQUIRES_DEPOSIT_AND_APPROVAL");

  const allowedToRun = taskType === "gpu"
    ? Boolean(job?.paid) && credit >= 99 && maxCost >= estimatedCost
    : Boolean(job?.paid) && maxCost >= estimatedCost;

  return {
    job_id: job?.job_id || "unknown",
    task_type: taskType,
    route: route.route,
    estimated_cost_cad: estimatedCost,
    credit_cad: credit,
    estimated_margin_cad: Math.max(0, Number((credit - estimatedCost).toFixed(4))),
    allowed_to_run: allowedToRun,
    privacy_level: privacy,
    risk_flags: riskFlags,
    notes: route.notes,
    pay_url: PRODUCT.stripePaymentLink,
    quoted_at: new Date().toISOString(),
  };
}

function extractCheckoutObject(eventOrOrder: any) {
  if (eventOrOrder?.data?.object && typeof eventOrOrder.data.object === "object") return eventOrOrder.data.object;
  return eventOrOrder;
}

function isPaidCheckout(obj: any): boolean {
  const paymentStatus = String(obj?.payment_status || "").toLowerCase();
  const status = String(obj?.status || "").toLowerCase();
  const paidFlag = obj?.paid === true;
  const amountTotal = obj?.amount_total || obj?.amount;
  return paidFlag || paymentStatus === "paid" || (status === "complete" && Boolean(amountTotal));
}

function productMatches(obj: any): boolean {
  const paymentLink = String(obj?.payment_link || obj?.payment_link_id || "");
  const metadata = typeof obj?.metadata === "object" && obj.metadata ? obj.metadata : {};
  return (
    paymentLink.includes(PRODUCT.stripePaymentLinkId) ||
    metadata.product_id === PRODUCT.stripeProductId ||
    metadata.price_id === PRODUCT.stripePriceId ||
    metadata.oae_product === "compute_relay_credit" ||
    obj?.price_id === PRODUCT.stripePriceId ||
    obj?.product_id === PRODUCT.stripeProductId ||
    obj?.product === PRODUCT.name
  );
}

async function checkoutToJob(eventOrOrder: any) {
  const obj = extractCheckoutObject(eventOrOrder);
  if (!isPaidCheckout(obj)) throw new Error("object_is_not_paid_or_complete");
  if (!productMatches(obj)) throw new Error("object_does_not_match_oae_compute_relay_credit");

  const metadata = typeof obj?.metadata === "object" && obj.metadata ? obj.metadata : {};
  const customerEmail = obj?.customer_email || obj?.customer_details?.email || metadata.customer_email || "unknown";
  const amountTotal = obj?.amount_total || obj?.amount || PRODUCT.priceCad * 100;
  const sourceId = obj?.id || obj?.payment_intent || (await sha256(obj)).slice(0, 16);

  const job: any = {
    job_id: metadata.job_id || `stripe-${sourceId}`,
    paid: true,
    credit_cad: Number((Number(amountTotal) / 100).toFixed(2)),
    customer_email: customerEmail,
    task_type: metadata.task_type || obj?.task_type || "text",
    task: metadata.task || obj?.task || "Run a small OAE Compute Relay text job. Customer did not provide task metadata yet.",
    output_format: metadata.output_format || obj?.output_format || "markdown",
    max_compute_cost_cad: Number(metadata.max_compute_cost_cad || obj?.max_compute_cost_cad || 0.25),
    privacy_level: metadata.privacy_level || obj?.privacy_level || "unknown",
    stripe_checkout_session: String(obj?.object || "").toLowerCase() === "checkout.session" ? obj?.id : obj?.stripe_checkout_session,
    stripe_payment_intent: obj?.payment_intent || obj?.stripe_payment_intent,
    stripe_payment_link_id: obj?.payment_link || obj?.payment_link_id || PRODUCT.stripePaymentLinkId,
    source_event_sha256: await sha256(eventOrOrder),
    created_at: new Date().toISOString(),
    delivery: {
      method: metadata.delivery_method || obj?.delivery_method || "manual",
      target: metadata.delivery_target || customerEmail,
    },
  };
  job.queue_record_sha256 = await sha256(job);
  return job;
}

function protocol() {
  return {
    protocol: "OAE_A2A_COMPUTE_RELAY",
    version: "0.1",
    product: PRODUCT,
    methods: [
      "GET /health",
      "GET /protocol",
      "POST /quote",
      "POST /a2a/quote",
      "POST /stripe/event-to-queue",
      "POST /stripe/webhook"
    ],
    note: "This gateway creates quote/queue records. Compute execution remains with HF Jobs relay.py.",
  };
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return json({ ok: true });

  const url = new URL(req.url);
  const path = url.pathname.replace(/^\/functions\/v1\/oae-compute-relay-gateway/, "") || url.pathname;

  if (req.method === "GET" && (path === "/" || path === "/health")) {
    return json({ status: "ok", service: "oae-compute-relay-gateway", product: PRODUCT.name, time: new Date().toISOString() });
  }

  if (req.method === "GET" && path === "/protocol") {
    return json(protocol());
  }

  if (req.method === "POST" && (path === "/quote" || path === "/a2a/quote")) {
    const body = await req.json().catch(() => ({}));
    const job = body?.job || body;
    return json({ status: "quoted", quote: quoteJob(job), protocol: "OAE_A2A_COMPUTE_RELAY" });
  }

  if (req.method === "POST" && (path === "/stripe/event-to-queue" || path === "/stripe/webhook")) {
    const auth = requireSharedSecret(req);
    if (auth) return auth;

    // TODO: If STRIPE_WEBHOOK_SECRET is configured, verify the Stripe-Signature header here.
    // Until then, this function converts event JSON into queue records but should not be treated
    // as authoritative payment verification unless called from a trusted connector/control plane.
    const body = await req.json().catch(() => null);
    if (!body) return json({ status: "error", reason: "invalid_json" }, 400);

    try {
      const payloads = Array.isArray(body) ? body : [body];
      const jobs = [];
      const errors = [];
      for (const item of payloads) {
        try {
          const job = await checkoutToJob(item);
          jobs.push(job);
        } catch (err) {
          errors.push({ error: String((err as Error).message || err), source_sha256: await sha256(item) });
        }
      }
      return json({ status: jobs.length ? "queue_records_created" : "no_jobs_created", jobs, errors, count: jobs.length });
    } catch (err) {
      return json({ status: "error", reason: String((err as Error).message || err) }, 400);
    }
  }

  return json({ status: "not_found", path, protocol: protocol() }, 404);
});

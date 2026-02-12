// MV3-safe background: uses webNavigation + DNR redirect.
// Calls local FastAPI (127.0.0.1:8000/predict) trained on YOUR dataset.

const API = "http://127.0.0.1:8000/predict";
const RISK_THRESHOLD = 0.6;
const RULE_ID = 55501;

// Clean any old dynamic rules at startup
chrome.runtime.onInstalled.addListener(async () => {
  await chrome.declarativeNetRequest.updateDynamicRules({ removeRuleIds: [RULE_ID] });
});
chrome.runtime.onStartup.addListener(async () => {
  await chrome.declarativeNetRequest.updateDynamicRules({ removeRuleIds: [RULE_ID] });
});

// Helper: check URL with API
async function checkUrl(url) {
  console.log("[Ref-TABMNet] Checking:", url);
  try {
    const resp = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });
    if (!resp.ok) throw new Error(`API ${resp.status}`);
    const data = await resp.json();
    console.log("[Ref-TABMNet] Model Response:", data);
    return data; // { label, score }
  } catch (e) {
    console.error("[Ref-TABMNet] API error:", e);
    return null;
  }
}

// Helper to extract real target from Google redirect 
function unwrapGoogleRedirect(url) {
  try {
    const u = new URL(url);
    if (u.hostname.includes("google.com") && u.searchParams.has("q")) {
      // Typical Gmail/Google redirect link
      return u.searchParams.get("q");
    }
    if (u.hostname.includes("mail.google.com") && u.searchParams.has("url")) {
      // Gmail internal redirect format
      return u.searchParams.get("url");
    }
  } catch (e) {
    console.warn("[Ref-TABMNet] Failed to unwrap redirect:", e);
  }
  return url; // fallback to original
}

// UPDATED Navigation listener 
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  const url = details.url || "";
  if (!url.startsWith("http")) return;

  console.log("[TEST] onBeforeNavigate triggered for:", url);

  //Force block everything for now (bypass ML)
  await chrome.storage.session.set({ lastBlockedUrl: url, lastScore: 0.99 });

  await chrome.declarativeNetRequest.updateDynamicRules({
    removeRuleIds: [999],
    addRules: [{
      id: 999,
      priority: 1,
      action: { type: "redirect", redirect: { url: chrome.runtime.getURL("interstitial.html") } },
      condition: { urlFilter: "*", resourceTypes: ["main_frame"] }
    }]
  });

  setTimeout(async () => {
    await chrome.declarativeNetRequest.updateDynamicRules({ removeRuleIds: [999] });
  }, 4000);
});

// Message API for popup/content to check explicit URL
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "check-url" && msg.url) {
    checkUrl(msg.url).then(sendResponse);
    return true; // keep port open
  }
});

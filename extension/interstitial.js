(async () => {
  const { lastBlockedUrl, lastScore } = await chrome.storage.session.get(["lastBlockedUrl", "lastScore"]);
  const el = document.getElementById("desc");
  if (lastBlockedUrl) {
    el.innerHTML = `Ref-TABMNet flagged this URL as risky (score=${(lastScore||0).toFixed(2)}):<br><br><code>${lastBlockedUrl}</code>`;
  }

  document.getElementById("goBack").onclick = () => history.back();
  document.getElementById("proceed").onclick = () => {
    if (lastBlockedUrl) chrome.tabs.create({ url: lastBlockedUrl });
    window.close();
  };
})();

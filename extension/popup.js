document.getElementById("check").onclick = () => {
  const url = document.getElementById("u").value;
  chrome.runtime.sendMessage({ type: "check-url", url }, (res) => {
    if (chrome.runtime.lastError) {
      document.getElementById("out").textContent = "Error: " + chrome.runtime.lastError.message;
      return;
    }
    document.getElementById("out").textContent = JSON.stringify(res || {error:"no response"}, null, 2);
  });
};

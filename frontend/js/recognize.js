(() => {
  const video      = document.getElementById('video');
  const canvas     = document.getElementById('canvas');
  const startBtn   = document.getElementById('start-camera');
  const captureBtn = document.getElementById('capture');
  const resultDiv  = document.getElementById('result');
  const geoStatus  = document.getElementById('geo-status');
  const threshLabel= document.getElementById('threshold-label');

  let geoPosition = null, geoError = null;

  // Load threshold from backend config
  fetch(`${BACKEND_URL}/api/config`)
    .then(r => r.json())
    .then(d => { if (threshLabel) threshLabel.textContent = `Similarity threshold: ${d.threshold}`; })
    .catch(() => {});

  function requestGeo() {
    if (!navigator.geolocation) { geoError = 'Geolocation not supported'; return; }
    navigator.geolocation.getCurrentPosition(
      pos => { geoPosition = pos; geoError = null; if (geoStatus) geoStatus.textContent = `Location: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`; },
      err => { geoError = err.message; if (geoStatus) geoStatus.textContent = `Location unavailable: ${err.message}`; },
      { enableHighAccuracy: true, timeout: 7000, maximumAge: 60000 }
    );
  }

  async function startCamera() {
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (e) {
      alert('Camera error: ' + e.message);
    }
  }

  async function captureAndRecognize() {
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);

    resultDiv.innerHTML = '<p class="hint">Recognizing…</p>';

    const payload = { image_data: canvas.toDataURL('image/jpeg', 0.92) };
    if (geoPosition?.coords) {
      const { latitude, longitude, accuracy } = geoPosition.coords;
      payload.latitude = latitude;
      payload.longitude = longitude;
      payload.location_label = `Browser location ±${Math.round(accuracy || 0)}m`;
    } else if (geoError) {
      payload.location_label = `Geolocation unavailable: ${geoError}`;
    }

    try {
      const resp = await fetch(`${BACKEND_URL}/api/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!data.ok) {
        resultDiv.innerHTML = `<div class="flash error">${data.error || data.message}</div>`;
        return;
      }
      if (data.match) {
        const p = data.person;
        resultDiv.innerHTML = `
          <div class="flash success">Match found! Score: ${data.score}</div>
          <div class="grid" style="margin-top:12px;">
            <div><img src="${p.image_url}" class="thumb" style="width:120px;height:120px;" alt="${p.name}"/></div>
            <div>
              <p><strong>Name:</strong> ${p.name}</p>
              <p><strong>Location:</strong> ${p.location || '—'}</p>
              <p><strong>Gender:</strong> ${p.gender || '—'}</p>
            </div>
          </div>`;
      } else {
        resultDiv.innerHTML = `<div class="flash error">No match found (best score: ${data.score}).</div>`;
      }
    } catch (err) {
      resultDiv.innerHTML = `<div class="flash error">Failed: ${err.message}</div>`;
    }
  }

  requestGeo();
  startBtn?.addEventListener('click', startCamera);
  captureBtn?.addEventListener('click', captureAndRecognize);
})();

(() => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const startBtn = document.getElementById('start-camera');
  const captureBtn = document.getElementById('capture');
  const resultDiv = document.getElementById('result');
  let geoPosition = null;
  let geoError = null;

  function requestGeolocation() {
    if (!navigator.geolocation) {
      geoError = 'Geolocation is not supported by this browser.';
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        geoPosition = pos;
        geoError = null;
      },
      (err) => {
        geoError = err.message;
      },
      {
        enableHighAccuracy: true,
        timeout: 7000,
        maximumAge: 60000,
      },
    );
  }

  async function startCamera() {
    try {
      requestGeolocation();
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
    } catch (err) {
      alert('Could not access camera: ' + err.message);
    }
  }

  function captureFrame() {
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, w, h);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    recognize(dataUrl);
  }

  async function recognize(dataUrl) {
    resultDiv.innerHTML = '<p class="hint">Recognizing...</p>';
    try {
      const payload = { image_data: dataUrl };
      if (geoPosition && geoPosition.coords) {
        const { latitude, longitude, accuracy } = geoPosition.coords;
        payload.latitude = latitude;
        payload.longitude = longitude;
        if (typeof accuracy === 'number') {
          payload.location_label = `Browser location ±${Math.round(accuracy)}m`;
        } else {
          payload.location_label = 'Browser location';
        }
      } else if (geoError) {
        payload.location_label = `Geolocation unavailable: ${geoError}`;
      }

      const resp = await fetch(RECOGNIZE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!data.ok) {
        resultDiv.innerHTML = `<p class="flash error">${data.error || data.message || 'Error'} </p>`;
        return;
      }
      if (data.match) {
        const p = data.person;
        resultDiv.innerHTML = `
          <div class="flash success">Match! Score: ${data.score}</div>
          <div class="grid">
            <div>
              <img src="${p.image_url}" alt="Person image" />
            </div>
            <div>
              <p><strong>Name:</strong> ${p.name}</p>
              <p><strong>Location:</strong> ${p.location || ''}</p>
              <p><strong>Gender:</strong> ${p.gender || ''}</p>
            </div>
          </div>
        `;
      } else {
        resultDiv.innerHTML = `<div class="flash error">No match (best score: ${data.score}).</div>`;
      }
    } catch (e) {
      resultDiv.innerHTML = `<p class="flash error">Failed: ${e.message}</p>`;
    }
  }

  requestGeolocation();
  startBtn?.addEventListener('click', startCamera);
  captureBtn?.addEventListener('click', captureFrame);
})();


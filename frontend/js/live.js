(() => {
  const streamImg  = document.getElementById('stream');
  const badge      = document.getElementById('pi-badge');
  const statusEl   = document.getElementById('stream-status');
  const geoStatus  = document.getElementById('live-location-status');

  // Generate a per-session token client-side
  const token = Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);

  const locationEndpoint = `${BACKEND_URL}/api/live_location/${token}`;
  const statusEndpoint   = `${BACKEND_URL}/api/pi_status`;
  const feedUrl          = `${BACKEND_URL}/video_feed?token=${token}`;

  // Load the MJPEG stream
  function loadStream() {
    streamImg.src = `${feedUrl}&_t=${Date.now()}`;
    if (statusEl) statusEl.textContent = '';
  }

  streamImg.addEventListener('error', () => {
    if (statusEl) statusEl.textContent = 'Stream interrupted — reconnecting in 8 s…';
    setTimeout(loadStream, 8000);
  });

  loadStream();

  // Poll Pi connection status
  let wasLive = false;
  function pollStatus() {
    fetch(statusEndpoint)
      .then(r => r.json())
      .then(data => {
        if (badge) {
          if (data.live) {
            badge.textContent = 'Pi Connected';
            badge.style.background = '#d1fae5';
            badge.style.color = '#065f46';
          } else {
            badge.textContent = 'Waiting for Pi…';
            badge.style.background = '#fef3c7';
            badge.style.color = '#92400e';
          }
        }
        if (data.live && !wasLive) loadStream();
        wasLive = data.live;
      })
      .catch(() => {
        if (badge) { badge.textContent = 'Server unreachable'; badge.style.background = '#fee2e2'; badge.style.color = '#991b1b'; }
      });
  }
  pollStatus();
  setInterval(pollStatus, 4000);

  // Geolocation → send to backend for detection log overlay
  let lastSent = 0;
  function sendLocation(payload) {
    if (Date.now() - lastSent < 5000) return;
    lastSent = Date.now();
    fetch(locationEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {});
  }

  if (!navigator.geolocation) {
    if (geoStatus) geoStatus.textContent = 'Geolocation not supported.';
  } else {
    navigator.geolocation.watchPosition(
      pos => {
        const { latitude, longitude, accuracy } = pos.coords;
        const label = `Browser location ±${Math.round(accuracy || 0)}m`;
        if (geoStatus) geoStatus.textContent = `Location: ${latitude.toFixed(4)}, ${longitude.toFixed(4)} (${label})`;
        sendLocation({ latitude, longitude, location_label: label });
      },
      err => {
        if (geoStatus) geoStatus.textContent = `Location unavailable: ${err.message}`;
        sendLocation({ location_label: `Geolocation unavailable: ${err.message}` });
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  }
})();

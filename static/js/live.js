(() => {
  const streamImg = document.getElementById('stream');
  const statusEl = document.getElementById('live-location-status');
  const config = window.liveConfig || {};
  if (!streamImg || !config.locationEndpoint) {
    return;
  }

  const updateStatus = (text) => {
    if (statusEl) {
      statusEl.textContent = text;
    }
  };

  let lastSent = 0;

  const sendUpdate = (payload) => {
    const now = Date.now();
    if (now - lastSent < 5000) {
      return;
    }
    lastSent = now;
    fetch(config.locationEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {
      /* ignore network errors */
    });
  };

  if (!navigator.geolocation) {
    updateStatus('Geolocation is not supported by this browser.');
    sendUpdate({ location_label: 'Geolocation unsupported' });
    return;
  }

  const handleSuccess = (pos) => {
    const { latitude, longitude, accuracy } = pos.coords;
    const label =
      typeof accuracy === 'number'
        ? `Browser location ±${Math.round(accuracy)}m`
        : 'Browser location';
    updateStatus(
      `Location active: ${latitude.toFixed(4)}, ${longitude.toFixed(4)} (${label})`,
    );
    sendUpdate({
      latitude,
      longitude,
      location_label: label,
    });
  };

  const handleError = (err) => {
    const message = `Geolocation unavailable: ${err.message}`;
    updateStatus(message);
    sendUpdate({ location_label: message });
  };

  navigator.geolocation.watchPosition(handleSuccess, handleError, {
    enableHighAccuracy: true,
    timeout: 10000,
    maximumAge: 0,
  });
})();

(() => {
  const video        = document.getElementById('video');
  const canvas       = document.getElementById('canvas');
  const imageData    = document.getElementById('image_data');
  const startBtn     = document.getElementById('start-camera');
  const captureBtn   = document.getElementById('capture');
  const captureStatus= document.getElementById('capture-status');
  const form         = document.getElementById('register-form');
  const alertEl      = document.getElementById('alert');

  function showAlert(msg, type = 'error') {
    alertEl.innerHTML = `<div class="flash ${type}">${msg}</div>`;
    alertEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  async function startCamera() {
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (e) {
      showAlert('Could not access camera: ' + e.message);
    }
  }

  function capturePhoto() {
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);
    imageData.value = canvas.toDataURL('image/jpeg', 0.92);
    if (captureStatus) captureStatus.textContent = '✓ Photo captured — click Register Person to save.';
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    alertEl.innerHTML = '';

    const fd = new FormData(form);
    // If no file chosen and no camera capture, show error
    const hasFile    = fd.get('image') && fd.get('image').size > 0;
    const hasCapture = fd.get('image_data');
    if (!hasFile && !hasCapture) {
      showAlert('Please upload a photo or capture one with your camera.');
      return;
    }
    if (hasFile && hasCapture) {
      // Prefer file upload; clear the base64 capture to avoid sending both
      fd.delete('image_data');
    }

    const btn = form.querySelector('#submit-btn');
    btn.disabled = true;
    btn.textContent = 'Registering…';

    try {
      const resp = await fetch(`${BACKEND_URL}/api/register`, { method: 'POST', body: fd });
      const data = await resp.json();
      if (data.ok) {
        showAlert(data.message || 'Registered successfully!', 'success');
        form.reset();
        imageData.value = '';
        if (captureStatus) captureStatus.textContent = '';
      } else {
        showAlert(data.error || 'Registration failed.');
      }
    } catch (err) {
      showAlert('Network error: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Register Person';
    }
  });

  startBtn?.addEventListener('click', startCamera);
  captureBtn?.addEventListener('click', capturePhoto);
})();

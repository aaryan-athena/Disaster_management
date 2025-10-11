(() => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const imageDataInput = document.getElementById('image_data');
  const startBtn = document.getElementById('start-camera');
  const captureBtn = document.getElementById('capture');

  async function startCamera() {
    try {
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
    imageDataInput.value = dataUrl;
  }

  startBtn?.addEventListener('click', startCamera);
  captureBtn?.addEventListener('click', captureFrame);
})();


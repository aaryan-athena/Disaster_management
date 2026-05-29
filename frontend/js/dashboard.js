(() => {
  const alertEl   = document.getElementById('alert');

  function fmt(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' });
  }

  function deleteBtn(id, name) {
    return `<button class="btn btn-danger" style="padding:6px 14px;font-size:13px;"
              onclick="deletePerson('${id}','${encodeURIComponent(name)}')">Delete</button>`;
  }

  window.deletePerson = async function(id, encodedName) {
    const name = decodeURIComponent(encodedName);
    if (!confirm(`Delete ${name}?`)) return;
    try {
      const resp = await fetch(`${BACKEND_URL}/api/persons/${id}`, { method: 'DELETE' });
      const data = await resp.json();
      if (data.ok) {
        alertEl.innerHTML = `<div class="flash success">Deleted ${name}.</div>`;
        loadDashboard();
      } else {
        alertEl.innerHTML = `<div class="flash error">${data.error || 'Delete failed.'}</div>`;
      }
    } catch (e) {
      alertEl.innerHTML = `<div class="flash error">Network error: ${e.message}</div>`;
    }
  };

  async function loadDashboard() {
    try {
      const resp = await fetch(`${BACKEND_URL}/api/dashboard`);
      const data = await resp.json();

      document.getElementById('stat-total').textContent    = data.summary.total_registered;
      document.getElementById('stat-detected').textContent = data.summary.with_detections;
      document.getElementById('stat-never').textContent    = data.summary.never_detected;

      // Detected table
      const dTbody = document.getElementById('detected-tbody');
      if (data.detected.length === 0) {
        dTbody.innerHTML = '<tr><td colspan="8" class="hint" style="text-align:center;padding:20px;">No detections yet.</td></tr>';
      } else {
        dTbody.innerHTML = data.detected.map(p => {
          const det = p.detection || {};
          const coords = det.latitude != null
            ? `${det.latitude.toFixed(4)}, ${det.longitude.toFixed(4)}` : '—';
          return `<tr>
            <td><div class="person-cell"><strong>${p.name}</strong><small>${p.id}</small></div></td>
            <td>${p.image_url ? `<img src="${p.image_url}" class="thumb" alt="${p.name}"/>` : '—'}</td>
            <td>${p.gender || '—'}</td>
            <td>${p.location || '—'}</td>
            <td>${fmt(det.last_seen_at)}</td>
            <td>${det.location || '—'}</td>
            <td class="muted">${coords}</td>
            <td>${deleteBtn(p.id, p.name)}</td>
          </tr>`;
        }).join('');
      }

      // Undetected table
      const uTbody = document.getElementById('undetected-tbody');
      if (data.undetected.length === 0) {
        uTbody.innerHTML = '<tr><td colspan="5" class="hint" style="text-align:center;padding:20px;">All registered people have been detected.</td></tr>';
      } else {
        uTbody.innerHTML = data.undetected.map(p => `<tr>
          <td><div class="person-cell"><strong>${p.name}</strong><small>${p.id}</small></div></td>
          <td>${p.gender || '—'}</td>
          <td>${p.location || '—'}</td>
          <td>${fmt(p.created_at)}</td>
          <td>${deleteBtn(p.id, p.name)}</td>
        </tr>`).join('');
      }

    } catch (e) {
      alertEl.innerHTML = `<div class="flash error">Failed to load dashboard: ${e.message}</div>`;
    }
  }

  loadDashboard();
})();

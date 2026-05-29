// Renders the shared navigation into <nav id="main-nav">.
(function () {
  const links = [
    { href: '/index.html',    label: 'Home',        aliases: ['/'] },
    { href: '/register.html', label: 'Register' },
    { href: '/recognize.html',label: 'Recognize' },
    { href: '/live.html',     label: 'Live' },
    { href: '/disaster.html', label: 'Disaster AI' },
    { href: '/dashboard.html',label: 'Dashboard' },
  ];

  const path = window.location.pathname;

  const html = links.map(({ href, label, aliases = [] }) => {
    const active = path === href || path === href.replace('.html', '') ||
                   aliases.some(a => path === a || path === a.replace('.html', ''));
    return `<a href="${href}"${active ? ' class="active"' : ''}>${label}</a>`;
  }).join('');

  const nav = document.getElementById('main-nav');
  if (nav) nav.innerHTML = html;
})();

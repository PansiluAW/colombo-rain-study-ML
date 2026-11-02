/* Progressive enhancement: the regular HTML form also works without JavaScript. */
(() => {
  const form = document.querySelector('#weather-form');
  if (!form) return;
  const panel = document.querySelector('#prediction');
  const status = document.querySelector('#update-status');
  const humidity = form.querySelector('#humidity_today');
  const sliderLabel = document.createElement('label');
  sliderLabel.className = 'slider-label';
  sliderLabel.textContent = 'Explore humidity';
  const slider = document.createElement('input');
  Object.assign(slider, {type: 'range', min: '0', max: '100', step: 'any', value: humidity.value});
  slider.setAttribute('aria-label', 'Explore humidity, percent');
  sliderLabel.append(slider);
  const scale = document.createElement('span');
  scale.className = 'slider-scale';
  scale.innerHTML = '<span>0% dry air</span><span>100% saturated</span>';
  sliderLabel.append(scale);
  humidity.closest('label').after(sliderLabel);
  const groups = [...form.querySelectorAll('.weather-group')];
  const readingsToggle = document.querySelector('#readings-toggle');
  if (readingsToggle) {
    function syncGroups() {
      const allOpen = groups.every(group => group.open);
      readingsToggle.textContent = allOpen ? 'Compact view' : 'Show all readings';
      readingsToggle.setAttribute('aria-expanded', String(allOpen));
    }
    readingsToggle.hidden = false;
    readingsToggle.addEventListener('click', () => {
      const expand = !groups.every(group => group.open);
      groups.forEach((group, i) => { group.open = expand || i === 0; });
      syncGroups();
    });
    groups.forEach(group => group.addEventListener('toggle', syncGroups));
    syncGroups();
  }
  form.addEventListener('invalid', event => {
    const group = event.target.closest('.weather-group');
    if (group) group.open = true;
  }, true);
  let timer, controller, revision = 0;
  async function update() {
    const current = ++revision;
    controller?.abort();
    if (!form.checkValidity()) {
      panel.querySelector('.prediction-body').innerHTML = '<div class="error" role="alert"><strong>Check the readings</strong><p>Complete the highlighted fields within their allowed ranges.</p></div>';
      status.textContent = 'Check inputs';
      panel.removeAttribute('aria-busy');
      return;
    }
    controller = new AbortController();
    panel.setAttribute('aria-busy', 'true');
    status.textContent = 'Calculating…';
    try {
      const response = await fetch(form.action.split('#')[0], {method: 'POST', body: new FormData(form), signal: controller.signal});
      if (![200, 422].includes(response.status)) throw new Error('Request failed');
      const page = new DOMParser().parseFromString(await response.text(), 'text/html');
      const content = page.querySelector('.prediction-body');
      if (!content || current !== revision) return;
      panel.querySelector('.prediction-body').replaceWith(content);
      document.querySelectorAll('.examples a').forEach((link, i) => link.classList.toggle('selected', page.querySelectorAll('.examples a')[i]?.classList.contains('selected')));
      status.textContent = response.ok ? 'Updated for your readings' : 'Check inputs';
    } catch (error) {
      if (error.name === 'AbortError' || current !== revision) return;
      status.textContent = 'Update failed';
      panel.querySelector('.prediction-body').innerHTML = '<div class="error" role="alert"><strong>Connection interrupted</strong><p>Use Calculate prediction to try again.</p></div>';
    } finally {
      if (current === revision) panel.removeAttribute('aria-busy');
    }
  }
  function schedule() {
    clearTimeout(timer);
    ++revision;
    controller?.abort();
    status.textContent = 'Readings changed…';
    timer = setTimeout(update, 600);
  }
  slider.addEventListener('input', () => {
    humidity.value = slider.value;
    schedule();
  });
  form.addEventListener('input', event => {
    if (event.target === slider) return;
    if (event.target === humidity) slider.value = humidity.value;
    schedule();
  });
  form.addEventListener('submit', event => {
    event.preventDefault();
    clearTimeout(timer);
    update();
    if (window.matchMedia('(max-width: 760px)').matches) panel.scrollIntoView({behavior: 'smooth'});
  });
})();

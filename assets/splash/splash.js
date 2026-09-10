/* SMIC splash — componente Streamlit (protocolo postMessage, sin npm). */
(function () {
  const DURATION_MS = 5000;
  const DEFAULT_VOLUME = 0.38;

  let started = false;
  let muted = false;
  let audio = null;
  let audioUnlocked = false;
  let reduceMotion = false;

  function send(type, data) {
    window.parent.postMessage(
      Object.assign({ isStreamlitMessage: true, type: type }, data || {}),
      "*"
    );
  }

  function setHeight() {
    send("streamlit:setFrameHeight", { height: window.innerHeight });
  }

  function setValue(value) {
    send("streamlit:setComponentValue", { value: value });
  }

  function pctLabel(n) {
    return "Carga " + n + "\u00a0%";
  }

  function easeOut(t) {
    return 1 - Math.pow(1 - t, 2.4);
  }

  function tryPlay() {
    if (!audio || muted || audioUnlocked) {
      return;
    }
    const p = audio.play();
    if (p && typeof p.then === "function") {
      p.then(function () {
        audioUnlocked = true;
        setHint("");
      }).catch(function () {
        setHint("Haga clic o pulse una tecla para habilitar el audio de entrada.");
      });
    }
  }

  function setHint(text) {
    const el = document.getElementById("hint");
    if (el) {
      el.textContent = text;
    }
  }

  function unlockAudio() {
    tryPlay();
  }

  function initParticles(canvas) {
    const ctx = canvas.getContext("2d");
    const dots = [];
    const n = 42;

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener("resize", resize);

    for (let i = 0; i < n; i++) {
      dots.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: 1.1 + Math.random() * 2.2,
        vx: (Math.random() - 0.5) * 0.22,
        vy: (Math.random() - 0.5) * 0.18,
        a: 0.08 + Math.random() * 0.14,
      });
    }

    function tick() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#047857";
      dots.forEach(function (d) {
        d.x += d.vx;
        d.y += d.vy;
        if (d.x < 0) d.x = canvas.width;
        if (d.x > canvas.width) d.x = 0;
        if (d.y < 0) d.y = canvas.height;
        if (d.y > canvas.height) d.y = 0;
        ctx.globalAlpha = d.a;
        ctx.beginPath();
        ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
      window.requestAnimationFrame(tick);
    }

    tick();
  }

  function runProgress(onDone) {
    const fill = document.getElementById("fill");
    const label = document.getElementById("pct");
    const t0 = performance.now();

    function frame(now) {
      const t = Math.min(1, (now - t0) / DURATION_MS);
      const eased = reduceMotion ? t : easeOut(t);
      const pct = Math.round(eased * 100);
      fill.style.width = pct + "%";
      label.textContent = pctLabel(pct);
      const bar = document.getElementById("bar");
      if (bar) {
        bar.setAttribute("aria-valuenow", String(pct));
      }
      if (t < 1) {
        window.requestAnimationFrame(frame);
      } else {
        fill.style.width = "100%";
        label.textContent = pctLabel(100);
        onDone();
      }
    }

    window.requestAnimationFrame(frame);
  }

  function start(args) {
    if (started) {
      return;
    }
    started = true;
    reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const audioSrc = args.audio_uri || "./intro-cue.wav";
    audio = new Audio(audioSrc);
    audio.preload = "auto";
    audio.volume = DEFAULT_VOLUME;
    audio.setAttribute("playsinline", "true");
    audio.addEventListener("error", function () {
      if (audioSrc !== "./intro-cue.wav") {
        audio = new Audio("./intro-cue.wav");
        audio.volume = DEFAULT_VOLUME;
        tryPlay();
      }
    });

    const muteBtn = document.getElementById("mute");
    muteBtn.addEventListener("click", function () {
      muted = !muted;
      if (audio) {
        audio.muted = muted;
        if (!muted) {
          tryPlay();
        }
      }
      muteBtn.setAttribute("aria-pressed", muted ? "true" : "false");
      muteBtn.textContent = muted ? "Activar sonido" : "Silenciar";
    });

    document.addEventListener("click", unlockAudio, { once: false });
    document.addEventListener("keydown", unlockAudio, { once: false });

    if (!reduceMotion) {
      initParticles(document.getElementById("bg"));
    }

    tryPlay();
    runProgress(function () {
      setValue({ done: true });
    });
  }

  window.addEventListener("message", function (event) {
    if (!event.data || event.data.type !== "streamlit:render") {
      return;
    }
    start(event.data.args || {});
  });

  window.addEventListener("resize", setHeight);
  send("streamlit:componentReady", { apiVersion: 1 });
  setHeight();
})();

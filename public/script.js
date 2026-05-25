/* ============================================
   Apple-Style Blog — JavaScript
   ============================================ */

(function () {
  'use strict';

  /* ---- Scroll-triggered Fade-In Animations ---- */

  function initScrollAnimations() {
    var animatedElements = document.querySelectorAll('.animate-in');
    if (!animatedElements.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.style.animationPlayState = 'running';
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
    );

    animatedElements.forEach(function (el) {
      el.style.animationPlayState = 'paused';
      observer.observe(el);
    });
  }

  /* ---- Reading Progress Bar ---- */

  function initProgressBar() {
    var progressBar = document.querySelector('.progress-bar');
    if (!progressBar) return;

    function updateProgress() {
      var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      var docHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      var progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      progressBar.style.width = Math.min(progress, 100) + '%';
    }

    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  /* ---- Smooth Scroll for Anchor Links ---- */

  function initSmoothScroll() {
    document.addEventListener('click', function (e) {
      var link = e.target.closest('a[href^="#"]');
      if (!link) return;

      var targetId = link.getAttribute('href');
      if (targetId === '#' || targetId === '#top') {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' });
        return;
      }

      var target = document.querySelector(targetId);
      if (!target) return;

      e.preventDefault();
      var navHeight = document.querySelector('.article-nav');
      var offset = navHeight ? navHeight.offsetHeight + 16 : 16;
      var targetPosition = target.getBoundingClientRect().top + window.pageYOffset - offset;

      window.scrollTo({ top: targetPosition, behavior: 'smooth' });

      if (history.pushState) {
        history.pushState(null, null, targetId);
      }
    });
  }

  /* ---- Mobile Navigation Toggle ---- */

  function initMobileNav() {
    var toggle = document.querySelector('.nav-toggle');
    var navMenu = document.querySelector('.nav-menu');
    if (!toggle || !navMenu) return;

    toggle.addEventListener('click', function () {
      toggle.classList.toggle('active');
      navMenu.classList.toggle('open');
      var isOpen = navMenu.classList.contains('open');
      toggle.setAttribute('aria-expanded', String(isOpen));
    });

    document.addEventListener('click', function (e) {
      if (!toggle.contains(e.target) && !navMenu.contains(e.target)) {
        toggle.classList.remove('active');
        navMenu.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  /* ---- Table of Contents Active State ---- */

  function initTocHighlight() {
    var tocLinks = document.querySelectorAll('.toc__list a');
    if (!tocLinks.length) return;

    var sections = [];
    tocLinks.forEach(function (link) {
      var id = link.getAttribute('href');
      if (id && id.startsWith('#')) {
        var section = document.querySelector(id);
        if (section) sections.push({ el: section, link: link });
      }
    });

    if (!sections.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            tocLinks.forEach(function (l) { l.classList.remove('active'); });
            var match = sections.find(function (s) { return s.el === entry.target; });
            if (match) match.link.classList.add('active');
          }
        });
      },
      { rootMargin: '-80px 0px -60% 0px', threshold: 0 }
    );

    sections.forEach(function (s) { observer.observe(s.el); });
  }

  /* ---- Nav Title Visibility on Scroll ---- */

  function initNavTitle() {
    var hero = document.querySelector('.article-hero');
    var navTitle = document.querySelector('.article-nav__title');
    if (!hero || !navTitle) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          navTitle.style.opacity = entry.isIntersecting ? '0' : '1';
        });
      },
      { threshold: 0 }
    );

    observer.observe(hero);
  }

  /* ---- Disable Copy / Cut / Context Menu ---- */

  function initCopyProtection() {
    document.addEventListener('copy', function (e) {
      e.preventDefault();
    });
    document.addEventListener('cut', function (e) {
      e.preventDefault();
    });
    document.addEventListener('contextmenu', function (e) {
      e.preventDefault();
    });
  }

  /* ---- Init ---- */

  function init() {
    initScrollAnimations();
    initProgressBar();
    initSmoothScroll();
    initMobileNav();
    initTocHighlight();
    initNavTitle();
    initCopyProtection();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

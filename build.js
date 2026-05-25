const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

const BLOG_DIR = __dirname;
const DIST_DIR = path.join(BLOG_DIR, 'dist');
const CONTENT_DIR = path.join(BLOG_DIR, 'content');
const TEMPLATE_DIR = path.join(BLOG_DIR, 'templates');
const PUBLIC_DIR = path.join(BLOG_DIR, 'public');

const isWatch = process.argv.includes('--watch');

function log(msg) {
  const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  console.log(`[${time}] ${msg}`);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function readFile(p) {
  return fs.readFileSync(p, 'utf-8');
}

function parseFrontMatter(src) {
  const match = src.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { meta: {}, content: src };
  const meta = {};
  for (const line of match[1].split('\n')) {
    const idx = line.indexOf(':');
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim();
    let val = line.slice(idx + 1).trim();
    if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1, -1);
    if (/^\d+$/.test(val)) val = parseInt(val, 10);
    meta[key] = val;
  }
  return { meta, content: match[2] };
}

function extractReferences(src) {
  const refSectionMatch = src.match(/\n## 参考文献\n\n([\s\S]*)$/);
  if (!refSectionMatch) return { body: src, referencesHtml: '' };

  const body = src.slice(0, refSectionMatch.index);
  const refLines = refSectionMatch[1].trim().split('\n\n');
  const items = [];
  for (const block of refLines) {
    const numMatch = block.match(/^\[(\d+)\]\s*/);
    if (!numMatch) continue;
    const num = numMatch[1];
    let text = block.slice(numMatch[0].length).trim();
    items.push({ num, text });
  }

  if (items.length === 0) return { body, referencesHtml: '' };

  const lis = items.map(({ num, text }) => {
    const html = inlineMarkdown(text);
    return `          <li data-ref="[${num}]" id="ref-${num}">${html}</li>`;
  }).join('\n');

  const html = `
      <section class="references" id="references">
        <details>
          <summary><h2>参考文献</h2></summary>
          <ol class="references__list">
${lis}
          </ol>
        </details>
      </section>`;

  return { body, referencesHtml: html };
}

function inlineMarkdown(text) {
  let s = text;
  s = s.replace(/&/g, '&amp;');
  s = s.replace(/</g, '&lt;');
  s = s.replace(/>/g, '&gt;');
  // Restore markdown syntax that was escaped
  s = s.replace(/&amp;/g, '&');
  // Bold
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Links [text](url)
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  // Inline code
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  return s;
}

function buildToc(html) {
  const headings = [];
  const regex = /<h([234])\s*(?:id="([^"]*)")?[^>]*>(.*?)<\/h\1>/g;
  let m;
  while ((m = regex.exec(html)) !== null) {
    const level = parseInt(m[1]);
    let id = m[2] || '';
    let text = m[3].replace(/<[^>]+>/g, '');
    if (!id) {
      id = text
        .replace(/[^\w\u4e00-\u9fff\s-]/g, '')
        .replace(/\s+/g, '-')
        .toLowerCase();
    }
    headings.push({ level, id, text });
  }

  if (headings.length === 0) return '';

  return headings.map(h => {
    const indent = h.level > 2 ? ' class="toc-indent"' : '';
    return `          <li${indent}><a href="#${h.id}">${h.text}</a></li>`;
  }).join('\n');
}

function processCustomHeadingIds(src) {
  // Convert `## Title {#custom-id}` to headings with id attribute
  return src.replace(/^(#{2,4})\s+(.+?)\s+\{#([^}]+)\}\s*$/gm, (match, hashes, title, id) => {
    const level = hashes.length;
    return `<h${level} id="${id}">${title}</h${level}>`;
  });
}

function processMath(src) {
  // Protect display math $$...$$ from markdown processing using raw HTML blocks
  // marked passes through raw HTML blocks untouched
  const displayMath = [];
  src = src.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
    const idx = displayMath.length;
    displayMath.push(math);
    // Use a raw HTML block marker that marked will pass through
    return `\n<div class="math-block" data-math-idx="${idx}"></div>\n`;
  });

  // Protect inline math $...$ (not $$) using HTML spans
  const inlineMath = [];
  src = src.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (match, math) => {
    const idx = inlineMath.length;
    inlineMath.push(math);
    return `<span class="math" data-math-idx="${idx}"></span>`;
  });

  return { src, displayMath, inlineMath };
}

function restoreMath(html, displayMath, inlineMath) {
  for (let i = 0; i < displayMath.length; i++) {
    const marker = '<div class="math-block" data-math-idx="' + i + '"></div>';
    const content = displayMath[i];
    html = html.replace(marker, function() {
      return '<div class="math-block">$$' + content + '$$</div>';
    });
  }
  for (let i = 0; i < inlineMath.length; i++) {
    const marker = '<span class="math" data-math-idx="' + i + '"></span>';
    const content = inlineMath[i];
    html = html.replace(marker, function() {
      return '<span class="math">$' + content + '$</span>';
    });
  }
  return html;
}

function processReferences(src) {
  // Convert [N] in text to <sup><a href="#ref-N">[N]</a></sup>
  // But not inside links or code blocks
  return src.replace(/(?<!\[)(?<!\()(?<!`)\[(\d+)\](?!\()(?!\))/g, (match, num) => {
    return `<sup><a href="#ref-${num}">[${num}]</a></sup>`;
  });
}

function markdownToHtml(src) {
  // Step 1: Process custom heading IDs (before marked)
  src = processCustomHeadingIds(src);

  // Step 2: Protect math expressions
  const { src: mathProtected, displayMath, inlineMath } = processMath(src);

  // Step 3: Process reference citations [N] → superscript links
  let withRefs = processReferences(mathProtected);

  // Step 4: Convert markdown to HTML using marked
  const renderer = new marked.Renderer();

  // Custom heading renderer to preserve IDs
  renderer.heading = function(text, level, raw) {
    const idMatch = raw.match(/\{#([^}]+)\}$/);
    let id = '';
    let cleanText = text;
    if (idMatch) {
      id = idMatch[1];
      cleanText = text.replace(/\{#[^}]+\}$/, '').trim();
    } else {
      id = cleanText
        .replace(/<[^>]+>/g, '')
        .replace(/[^\w\u4e00-\u9fff\s-]/g, '')
        .replace(/\s+/g, '-')
        .toLowerCase();
    }
    return `<h${level} id="${id}">${cleanText}</h${level}>`;
  };

  marked.setOptions({
    renderer,
    gfm: true,
    breaks: false,
  });

  let html = marked.parse(withRefs);

  // Step 5: Restore math expressions
  html = restoreMath(html, displayMath, inlineMath);

  // Step 6: Fix heading IDs for custom ID headings that were processed before marked
  // The custom heading tags were already converted to <hN id="..."> before marked,
  // but marked may have wrapped them in <p> tags. Clean up.
  html = html.replace(/<p><(h[234]\s+id="[^"]*">.*?<\/h[234]>)<\/p>/g, '<$1>');

  return html;
}

function buildChapter(chapterNum, chaptersData) {
  const chapterInfo = chaptersData.chapters.find(c => c.number === chapterNum);
  if (!chapterInfo) return null;

  const mdPath = path.join(CONTENT_DIR, chapterInfo.file);
  if (!fs.existsSync(mdPath)) {
    log(`  ⚠ Missing: ${mdPath}`);
    return null;
  }

  const raw = readFile(mdPath);
  const { meta, content } = parseFrontMatter(raw);
  const { body, referencesHtml } = extractReferences(content);

  // Build HTML content from markdown body
  const contentHtml = markdownToHtml(body);

  // Build TOC from the generated HTML
  const toc = buildToc(contentHtml);

  // Build nav links
  const num = chapterNum;
  const total = chaptersData.chapters.length;
  let navLinks = '';

  if (num > 1) {
    const prev = chaptersData.chapters.find(c => c.number === num - 1);
    navLinks += `      <a href="chapter-${num - 1}.html" class="article-pager__link">
        <span class="article-pager__label">上一章</span>
        <span class="article-pager__title">${prev.title}</span>
      </a>\n`;
  }
  if (num < total) {
    const next = chaptersData.chapters.find(c => c.number === num + 1);
    navLinks += `      <a href="chapter-${num + 1}.html" class="article-pager__link article-pager__link--next">
        <span class="article-pager__label">下一章</span>
        <span class="article-pager__title">${next.title}</span>
      </a>`;
  }

  if (navLinks) {
    navLinks = `      <nav class="article-pager">\n${navLinks}\n      </nav>`;
  }

  // Read template
  const template = readFile(path.join(TEMPLATE_DIR, 'article.html'));

  // Build meta string
  const readingTime = meta.readingTime || chapterInfo.readingTime;
  const date = meta.date || chapterInfo.date || '';
  const metaStr = `程乾 · ${date} · 约 ${readingTime} 分钟`;

  // Fill template
  let html = template
    .replace(/\{\{TITLE\}\}/g, meta.title || chapterInfo.title)
    .replace(/\{\{NAV_TITLE\}\}/g, meta.navTitle || chapterInfo.title)
    .replace(/\{\{CHAPTER_NUMBER\}\}/g, String(chapterNum))
    .replace(/\{\{FULL_TITLE\}\}/g, meta.fullTitle || chapterInfo.fullTitle)
    .replace(/\{\{META\}\}/g, metaStr)
    .replace(/\{\{TOC\}\}/g, toc)
    .replace(/\{\{CONTENT\}\}/g, function() { return contentHtml; })
    .replace(/\{\{REFERENCES\}\}/g, function() { return referencesHtml; })
    .replace(/\{\{NAV_LINKS\}\}/g, navLinks);

  return html;
}

function buildIndex(chaptersData) {
  const template = readFile(path.join(TEMPLATE_DIR, 'index.html'));

  const cards = chaptersData.chapters.map((ch, i) => {
    const delay = (i % 4) + 1;
    return `      <a href="chapter-${ch.number}.html" class="chapter-card animate-in animate-delay-${delay}">
        <span class="chapter-card__number">${ch.number}</span>
        <h2 class="chapter-card__title">${ch.title}</h2>
        <p class="chapter-card__excerpt">${ch.excerpt}</p>
        <p class="chapter-card__meta">约 ${ch.readingTime} 分钟阅读</p>
      </a>`;
  }).join('\n\n');

  return template
    .replace(/\{\{TITLE\}\}/g, chaptersData.title)
    .replace(/\{\{SUBTITLE\}\}/g, chaptersData.subtitle)
    .replace(/\{\{AUTHOR\}\}/g, chaptersData.author)
    .replace(/\{\{DATE\}\}/g, chaptersData.date)
    .replace(/\{\{CHAPTERS\}\}/g, cards);
}

function copyPublicAssets() {
  if (!fs.existsSync(PUBLIC_DIR)) {
    log('  ⚠ No public/ directory found, skipping assets');
    return;
  }
  const files = fs.readdirSync(PUBLIC_DIR);
  for (const file of files) {
    const src = path.join(PUBLIC_DIR, file);
    const dest = path.join(DIST_DIR, file);
    fs.copyFileSync(src, dest);
    log(`  ✓ ${file}`);
  }
}

function build() {
  const startTime = Date.now();
  log('Building...');

  ensureDir(DIST_DIR);

  // Read chapter metadata
  const chaptersData = JSON.parse(readFile(path.join(CONTENT_DIR, 'chapters.json')));

  // Build index
  log('  Building index.html');
  const indexHtml = buildIndex(chaptersData);
  fs.writeFileSync(path.join(DIST_DIR, 'index.html'), indexHtml, 'utf-8');
  log('  ✓ index.html');

  // Build each chapter
  for (const ch of chaptersData.chapters) {
    log(`  Building chapter-${ch.number}.html`);
    const html = buildChapter(ch.number, chaptersData);
    if (html) {
      fs.writeFileSync(path.join(DIST_DIR, `chapter-${ch.number}.html`), html, 'utf-8');
      log(`  ✓ chapter-${ch.number}.html`);
    }
  }

  // Copy public assets
  log('  Copying public assets');
  copyPublicAssets();

  const elapsed = Date.now() - startTime;
  log(`Build complete in ${elapsed}ms → dist/`);
}

function watch() {
  log('Watching for changes... (Ctrl+C to stop)');
  build();

  const watchDirs = [CONTENT_DIR, TEMPLATE_DIR, PUBLIC_DIR];
  const debounceMap = new Map();

  for (const dir of watchDirs) {
    if (!fs.existsSync(dir)) continue;
    fs.watch(dir, { recursive: true }, (eventType, filename) => {
      if (!filename) return;
      if (debounceMap.has(filename)) clearTimeout(debounceMap.get(filename));
      debounceMap.set(filename, setTimeout(() => {
        debounceMap.delete(filename);
        log(`Changed: ${filename}`);
        try {
          build();
        } catch (err) {
          console.error('Build error:', err.message);
        }
      }, 200));
    });
  }

  // Also watch build.js itself
  fs.watch(path.join(BLOG_DIR, 'build.js'), () => {
    log('build.js changed — restart required');
  });
}

if (isWatch) {
  watch();
} else {
  build();
}

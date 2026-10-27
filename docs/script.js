const copyButton = document.getElementById('copy-citation');
const citation = document.getElementById('bibtex');
const status = document.getElementById('copy-status');

if (navigator.clipboard && copyButton && citation && status) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent.trim() + '\n');
      status.textContent = 'BibTeX copied to clipboard.';
    } catch {
      status.textContent = 'Could not copy. Select the citation text or download the .bib file.';
    }
  });
}

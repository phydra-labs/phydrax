window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  chtml: {
    mtextInheritFont: true,
    merrorInheritFont: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|md-typeset"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});

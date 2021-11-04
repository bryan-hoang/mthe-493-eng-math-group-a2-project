\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mthe-report}[2021/10/03 A LaTeX class to write MTHE 493 reports]

% Pass down options to report class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax

\LoadClass[
  12pt,
  oneside,
  letterpaper,
]{memoir}

% Remove space before chapters and sections.
\setlength{\cftbeforechapterskip}{0pt plus 0pt}
\renewcommand*{\insertchapterspace}{}
\setlength{\cftbeforesectionskip}{0pt plus 0pt}

% Paragraph indentation and spacing
\setlength{\parindent}{0em}
\setlength{\parskip}{1em}
\checkandfixthelayout[nearest]

% Change running header for non-chapter pages.
\addtopsmarks{headings}{}{
  \createmark{chapter}{both}{shownumber}{}{. \ }
}
\pagestyle{headings}

% Changing the title of the table of contents section.
\RequirePackage[english]{babel}

\addto\captionsenglish{\renewcommand*\contentsname{Table Of Contents}}

% Change formatting of chapter titles
\RequirePackage{titlesec}

\titleformat{\chapter}{\normalfont\bfseries\huge}{\thechapter}{0.75em}{}
\titlespacing*{\chapter}{0pt}{0pt}{24pt}

% Number subsections and above.
\setsecnumdepth{subsection}
\maxtocdepth{subsection}

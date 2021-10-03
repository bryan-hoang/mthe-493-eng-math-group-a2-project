\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mthe-report}[2021/10/03 A LaTeX class to write MTHE 493 reports]

% Pass down options to report class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\LoadClass[
    11pt,
    letterpaper,
    titlepage,
    english,
]{report}

% Changing the title of the table of contents section.
\RequirePackage[english]{babel}

\addto\captionsenglish{\renewcommand*\contentsname{Table Of Contents}}

% Change formatting of chapter titles
\RequirePackage{titlesec}

\titleformat{\chapter}[hang]{\normalfont\bfseries}{\thechapter}{1em}{}
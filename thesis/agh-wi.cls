%%
%% This is file `agh-wi.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% agh-wi.dtx  (with options: `class')
%% ----------------------------------------------------------------
%% agh-wi --- Diploma thesis template for students of the Faculty of Computer Science of the AGH University of Krakow
%% email: polak[aT]agh[DoT]edu[DoT]pl
%% GitHub: https://github.com/polaksta
%% Released under the LaTeX Project Public License v1.3c or later
%% See http://www.latex-project.org/lppl.txt
%% ----------------------------------------------------------------
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\fileversion{1.3.2}
\def\filedate{2024/06/13}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{agh-wi}[\filedate\space v\fileversion\space Diploma thesis at WI AGH Kraków, Poland]

\typeout{..................................................................}
\typeout{You are using the 'agh-wi' class created by Stanisław Polak}
\typeout{Version: \fileversion, \filedate}
\typeout{..................................................................}
\newif\ifenglish
\newif\ifprint
\def\fieldOfStudy{Informatyka}
\DeclareOption{english}{\englishtrue}
\DeclareOption{print}{\printtrue}
\DeclareOption{ds}{\def\fieldOfStudy{Informatyka --- Data Science}}
\DeclareOption{umisi}{\def\fieldOfStudy{Informatyka --- Uczenie Maszynowe i Sztuczna Inteligencja}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}
\ProcessOptions\relax
\LoadClass[appendixprefix=true, numbers=enddot]{scrbook}
\RequirePackage{tikz}
\RequirePackage{scrlayer-scrpage}
\ifprint
    \KOMAoptions{fontsize=12pt, DIV=14, BCOR=2cm, automark, headsepline, footsepline, plainfootsepline,toc=listof, toc=bib}
\else
    \KOMAoptions{fontsize=12pt, DIV=14, automark, headsepline, footsepline, plainfootsepline, toc=listof, toc=bib}
\fi
\ifenglish
    \RequirePackage[main=english,polish]{babel}
\else
    \RequirePackage[main=polish,english]{babel}
\fi
\RequirePackage{indentfirst}
\AtBeginDocument{
    \@ifpackageloaded{hyperref}{
        \ifprint
            \hypersetup{hidelinks}
        \else
            \hypersetup{
                colorlinks=true,
                linktocpage=true,
                linkcolor=blue,
                citecolor=black,
                filecolor=magenta,
                urlcolor=cyan
            }
        \fi
    }{}
    \@ifpackageloaded{algorithm2e}{
        \ifenglish\else
            \renewcommand{\listalgorithmcfname}{Lista algorytmów}%
            \renewcommand{\algorithmcfname}{Algorytm}%
        \fi
    }{}
    \@ifpackageloaded{minted}{
        \@ifpackagewith{minted}{newfloat}{
            \ifenglish\else
                \SetupFloatingEnvironment{listing}{name=Kod źródłowy}
                \SetupFloatingEnvironment{listing}{listname=Lista kodów źródłowych}
            \fi
            \setminted{
                linenos,
                frame=single
            }
        }
        {
            \@latex@error{The 'minted' package must be loaded with the 'newfloat' option}\@ehc
        }
    }{}
    \@ifpackageloaded{listings}{
        \lstset{
            showstringspaces=false,
            numbers=left,
            stepnumber=1,
            numbersep=5pt,
            showspaces=false,
            showtabs=false,
            frame=single,
            tabsize=2,
            captionpos=t,
            breaklines=true,
            breakatwhitespace=false,
            keywordstyle=\ttfamily\color{blue},
            identifierstyle=\ttfamily\color{violet}\bfseries,
            commentstyle=\color{brown},
            stringstyle=\ttfamily,
            inputencoding=utf8,
            literate={ą}{{\k{a}}}1
                     {Ą}{{\k{A}}}1
                     {ę}{{\k{e}}}1
                     {Ę}{{\k{E}}}1
                     {ó}{{\'o}}1
                     {Ó}{{\'O}}1
                     {ś}{{\'s}}1
                     {Ś}{{\'S}}1
                     {ł}{{\l{}}}1
                     {Ł}{{\L{}}}1
                     {ż}{{\.z}}1
                     {Ż}{{\.Z}}1
                     {ź}{{\'z}}1
                     {Ź}{{\'Z}}1
                     {ć}{{\'c}}1
                     {Ć}{{\'C}}1
                     {ń}{{\'n}}1
                     {Ń}{{\'N}}1
    }
    \ifenglish
        \renewcommand{\lstlistlistingname}{List of Listings}
    \else
        \renewcommand{\lstlistingname}{Kod źródłowy}%
        \renewcommand{\lstlistlistingname}{Lista kodów źródłowych}
    \fi
    }{}
}
\definecolor{AGH@black}{RGB}{30,30,30}
\definecolor{AGH@red}{RGB}{167,25,48}
\definecolor{AGH@green}{RGB}{0,105,60}

\def\@titlePL{\@latex@error{No \noexpand\titlePL given}\@ehc}
\def\@titleEN{\@latex@error{No \noexpand\titleEN given}\@ehc}
\def\@author{\@latex@error{No \noexpand\author given}\@ehc}
\def\@supervisor{\@latex@error{No \noexpand\supervisor given}\@ehc}

\ifenglish
    \title{\@titleEN}
\else
    \title{\@titlePL}
\fi

\newcommand{\supervisor}[1]{\gdef\@supervisor{#1}}
\newcommand{\titlePL}[1]{\gdef\@titlePL{#1}}
\newcommand{\titleEN}[1]{\gdef\@titleEN{#1}}
\renewcommand\maketitle{
    \KOMAoptions{twoside = false}
    \begin{titlepage}
        \centering
         \begin{figure}[!ht]
            \begin{center}
                \includegraphics[width=0.3\linewidth]{img/pk.jpg}
            \end{center}
        \end{figure}\par
        
        \vspace*{3\baselineskip}%
        {    \bfseries\large
            Politechnika Krakowska im. Tadeusza Kościuszki
        }\par
        \vspace*{\baselineskip}%
        {
            \large Wydział Informatyki i Telekomunikacji
        }\par
        \vspace*{3\baselineskip}%
        {
            \scshape\Large Praca magisterska
        }\par
        \vspace*{3\baselineskip}%
        {\sffamily\Large\bfseries \@title}\par%
        \vspace*{\baselineskip}%
        \ifenglish
        {\selectlanguage{polish}\sffamily\small\bfseries \@titlePL}\par%
        \else
        {\selectlanguage{english}\sffamily\small\bfseries \@titleEN}\par%
        \fi
        \vfill
        \par
        \begin{tabular}{ll}
            Autor:         & {\bfseries
            \begin{tabular}[t]{c}
                        \@author
                    \end{tabular}}        \\[0.8mm]
            Kierunek:      & {\bfseries
            \begin{tabular}[t]{c}
                        \fieldOfStudy
                    \end{tabular}}        \\[0.8mm]
            Opiekun pracy: & {\bfseries
            \begin{tabular}[t]{c}
                        \@supervisor
                    \end{tabular}}        \\[0.8mm]
        \end{tabular}
        \par
        \vspace*{3\baselineskip}%
        \centering Kraków, \the\year
    \end{titlepage}%
    \KOMAoptions{twoside = true}
}
\newenvironment{abstractPL}{%
    \titlepage
    \null\vfil
    \@beginparpenalty\@lowpenalty
    \begin{center}%
        \bfseries Streszczenie
        \@endparpenalty\@M
    \end{center}}%
{\par\vfil\null\endtitlepage}
\newenvironment{abstractEN}{%
    \titlepage
    \null\vfil
    \@beginparpenalty\@lowpenalty
    \begin{center}%
        \bfseries Abstract
        \@endparpenalty\@M
    \end{center}}%
{\par\vfil\null\endtitlepage}
\newcommand*{\tocchapapp}{}
\newcommand*{\appendixmore}{
    \addtocontents{\csname ext@toc\endcsname}{
        \string\DeclareTOCStyleEntry[numwidth+=5em,
            entrynumberformat=\string\appendixprefixformat]{chapter}{chapter}
    }
}
\newcommand*{\appendixprefixformat}[1]{\appendixname\space#1}
%% 
%% Copyright (C) 2024 by Stanisław Polak <polak[aT]agh[DoT]edu[DoT]pl>
%% 
%% This work may be distributed and/or modified under the conditions of the LaTeX
%% Project Public License (LPPL), either version 1.3c of this license or (at your
%% option) any later version. The latest version of this license is in the file:
%% 
%% http://www.latex-project.org/lppl.txt
%% 
%% This work is "maintained" (as per LPPL maintenance status) by
%% Stanisław Polak.
%% This work consists of the file
%% agh-wi.dtx
%% and the derived files
%% agh-wi.ins,
%% agh-wi.cls,
%% agh-wi.pdf,
%% bibliography.bib, and
%% example.tex.
%% 
%%
%% End of file `agh-wi.cls'.

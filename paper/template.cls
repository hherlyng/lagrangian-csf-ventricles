\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{template}

\LoadClass[11pt]{article}
\RequirePackage[utf8]{inputenc}
\RequirePackage{graphicx}
\RequirePackage[font=small]{caption}
\RequirePackage[left=2cm,%
                right=2cm,%
                top=2.25cm,%
                bottom=2.25cm,%
                headheight=12pt,%
                a4paper]{geometry}%
\RequirePackage{textcomp, gensymb}
\RequirePackage{siunitx} % \num{} to format powers of 10
\RequirePackage{float} % Figure placement
\RequirePackage{amsmath} % Math
\RequirePackage{amssymb} % Blackboard letters
\RequirePackage{amsfonts}
\RequirePackage{amsopn}
\RequirePackage{mathtools}
\RequirePackage{enumerate}
\RequirePackage{booktabs} % Tables
\RequirePackage{color} % Text color
\RequirePackage{bm} % Bold math
\RequirePackage{authblk}
\RequirePackage{silence}
\RequirePackage{physics}
\RequirePackage{stmaryrd} % Double brackets for jump operator
\RequirePackage[pdftex, bookmarks, colorlinks, breaklinks, allcolors=blue]{hyperref} % Links
\RequirePackage{soul}
\RequirePackage{color}
\RequirePackage{xcolor,colortbl}
\hypersetup{colorlinks, citecolor=blue}
\RequirePackage[nameinlink, capitalise]{cleveref} % Cross referencing
\RequirePackage[superscript,ref]{cite} % Citation
\RequirePackage{csquotes} % Bibliography tools
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\WarningFilter*{latex}{Text page \thepage\space contains only floats}

% Author block
\setlength{\affilsep}{1.5em}
\renewcommand\Affilfont{\small} 

\newcommand{\threesubsection}[1]{
\medskip
\textit{#1}: 
}

% Bracket citations
\makeatletter
\renewcommand\@citess[1]{\textsuperscript{#1}} 
\makeatother

% Add panels to figure references
\newcommand{\Crefpanels}[2]{\hyperref[#1]{\Cref*{#1}#2}}

%% -----  Some custom commands  ----- %%

%% EDITING: REMOVE THESE BEFORE SUBMISSION
\newcommand{\lyng}[1]{\textcolor{blue}{HH: #1}}
\newcommand{\fixme}[1]{\textcolor{red}{#1}}

% Math
\newcommand{\pdifft}[1]{\frac{\partial  #1}{\partial t}} % Time derivative
\newcommand{\pdiffdifft}[1]{\frac{\partial^2 #1}{\partial t^2}}  % Second derivate

% Bold faced math characters
\newcommand{\cc}{\bm{c}}
\newcommand{\nn}{\bm{n}}
\newcommand{\rr}{\bm{r}}
%\newcommand{\qq}{\bm{q}}
\newcommand{\uu}{\bm{u}}
\newcommand{\vv}{\bm{v}}
\newcommand{\ww}{\bm{w}}
\newcommand{\xx}{\bm{x}}
\newcommand{\yy}{\bm{y}}
\newcommand{\JJ}{\bm{J}}
\newcommand{\VV}{\bm{V}}
\newcommand{\bsig}{\bm{\sigma}} % Bold sigma (stress tensor)
\newcommand{\bsigpar}{\hat{\bsig}_{\parallel}} % Tangential traction
\newcommand{\bsigperp}{\hat{\bsig}_{\perp}} % Normal traction
\newcommand{\beps}{\bm{\varepsilon}} % Bold epsilon (symmetric gradient)
\newcommand{\btau}{\bm{\tau}} % Bold tau (cilia traction)

% Bold faced math characters
\newcommand{\TT}{\mathbb{T}}
\newcommand{\DD}{\mathcal{D}}
\newcommand{\Scal}{\mathcal{S}}
\newcommand{\Bcal}{\mathcal{B}}
\newcommand{\bsigma}{\bm{\sigma}}
\newcommand{\bsigmahat}{\hat{\bm{\sigma}}}
\newcommand{\bphi}{\bm{\phi}}


% Discontinuity operators
\newcommand{\avg}[1]{\{#1\}}
\newcommand{\jump}[1]{\llbracket#1\rrbracket}

% Boundaries
\newcommand{\Gs}{\Gamma_{\mathrm{s}}} % Slip boundary
\newcommand{\Gc}{\Gamma_{\mathrm{c}}} % Cilia boundary
\newcommand{\Gp}{\Gamma_{\mathrm{p}}} % Anterior/Posterior pressure boundary
\newcommand{\Gnf}{\Gamma_{\mathrm{nf}}} % No-flux boundary
\newcommand{\Gin}{\Gamma_{\mathrm{in}}} % Inflow boundary
\newcommand{\Gout}{\Gamma_{\mathrm{out}}} % Outflow boundary

% Integrals
\newcommand{\dx}{\, \mathrm d\bm{x}}

% Norms
\newcommand{\normltwo}[1]{{\vert\vert#1\vert\vert}_{L^2}}
\newcommand{\normltwovec}[1]{{\vert\vert#1\vert\vert}_{\mathbf{L}^2}}
\newcommand{\normlinf}[1]{{\vert\vert#1\vert\vert}_{L^{\infty}}}
\newcommand{\Hnorm}[1]{\vert#1\vert_{\mathbf{H}^1}}
\newcommand{\Lnorm}[1]{\norm{#1}_{L^2}}
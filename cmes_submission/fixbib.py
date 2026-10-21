import pathlib, re
p = pathlib.Path(r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\cmes_submission\main.tex")
t = p.read_text(encoding="utf-8")

new_bib = r"""
\begin{thebibliography}{99}
\bibitem{born_lit}
Habashy TM, Groom RW, Spies BR. Beyond the Born and Rytov approximations: A nonlinear approach to electromagnetic scattering. J Geophys Res. 1993;98(B2):1759--1775.
\bibitem{csi_lit}
van den Berg PM, Kleinman RE. A contrast source inversion method. Inverse Probl. 1997;13(6):1607--1620.
\bibitem{som_lit}
Chen X. Subspace-based optimization method for solving inverse-scattering problems. IEEE Trans Geosci Remote Sens. 2010;48(1):42--49.
\bibitem{somnet}
Liu Y, Zhao H, Song R, Chen X, Li C, Chen X. SOM-Net: Unrolling the subspace-based optimization for solving full-wave inverse scattering problems. IEEE Trans Geosci Remote Sens. 2022;60:2007715.
\bibitem{unrolledcnn}
Zhang Y, Lambert M, Fraysse A, Lesselier D. Unrolled convolutional neural network for full-wave inverse scattering. IEEE Trans Antennas Propag. 2023;71(1):947--956.
\bibitem{neuralbim}
Shan T, Lin Z, Song X, Li M, Yang F, Xu S. Neural Born iterative method for solving inverse scattering problems: 2D cases. IEEE Trans Antennas Propag. 2023;71(1):818--829.
\bibitem{deepnis}
Li L, Wang LG, Teixeira FL, Liu C, Nehorai A, Cui TJ. DeepNIS: Deep neural network for nonlinear electromagnetic inverse scattering. IEEE Trans Antennas Propag. 2019;67(3):1819--1825.
\bibitem{mfneuralbim}
Liu D, Shan T, Li M, Yang F, Xu S. Multi-frequency neural Born iterative method for solving 2-D inverse scattering problems. IEEE Trans Comput Imaging. 2025;11:1243--1257.
\bibitem{ciesomnet}
Xu K, et al. Physically unrolling network under contraction integral equation for limited-aperture inverse scattering problem. IEEE Trans Antennas Propag. 2023;71(11):9130--9135.
\bibitem{kernelunroll}
Li X, Zhang T, Bi D, Peng L, Xie Y. Kernelized deep unfolding network with joint physical constraints for electromagnetic inverse scattering imaging. IEEE Trans Antennas Propag. 2026;74(2):1707--1722.
\bibitem{li2019cnn}
Li L, Wang LG, Teixeira FL. Performance analysis and dynamic evolution of deep convolutional neural network for electromagnetic inverse scattering. IEEE Antennas Wireless Propag Lett. 2019;18(11):2259--2263.
\bibitem{lista}
Gregor K, LeCun Y. Learning fast approximations of sparse coding. In: Proc ICML; 2010. p. 399--406.
\bibitem{listacp}
Chen X, Liu J, Wang Z, Yin W. Theoretical linear convergence of unfolded ISTA and its practical weights and thresholds. In: Proc NeurIPS; 2018.
\bibitem{alista}
Liu J, Chen X, Wang Z, Yin W. ALISTA: Analytic weights are as good as learned weights in LISTA. In: Proc ICLR; 2019.
\bibitem{cui_reg}
Cui ZX, Zhu Q, Cheng J, Zhang B, Liang D. Deep unfolding as iterative regularization for imaging inverse problems. Inverse Probl. 2024;40(2):025011.
\bibitem{sherry_nonexp}
Sherry F, Celledoni E, Ehrhardt MJ, Murari D, Owren B, Sch\"onlieb CB. Designing stable neural networks using convex analysis and ODEs. Physica D. 2024;463:134159.
\bibitem{mondeq}
Winston E, Kolter JZ. Monotone operator equilibrium networks. In: Proc NeurIPS; 2020.
\bibitem{algomapping}
Tay DB. On iterative algorithms with different mapping in each iteration. Algorithms. 2026;19(6):470.
\bibitem{mukherjee2023}
Mukherjee S, Hauptmann A, \"Oktem O, Pereyra M, Sch\"onlieb CB. Learned reconstruction methods with convergence guarantees: A survey of concepts and applications. IEEE Signal Process Mag. 2023;40(1):164--182.
\bibitem{monga_survey}
Monga V, Li Y, Eldar YC. Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing. IEEE Signal Process Mag. 2021;38(2):18--44.
\bibitem{convergentborn}
Kleinman RE, Roach GF, van den Berg PM. Convergent Born series for large refractive indices. J Opt Soc Am A. 1990;7(5):890--897.
\bibitem{cie_lit}
Isernia T, Crocco L, D'Urso M. New tools and series for forward and inverse scattering problems in lossy media. IEEE Geosci Remote Sens Lett. 2004;1(4):327--331.
\bibitem{kleinvandenberg}
Kleinman RE, van den Berg PM. Iterative methods for solving integral equations. Prog Electromagn Res. 1991;5:67--102.
\bibitem{slaney}
Slaney M, Kak AC, Larsen LE. Limitations of imaging with first-order diffraction tomography. IEEE Trans Microw Theory Tech. 1984;32(8):860--874.
\bibitem{devalle2023}
de Valle C, Centofanti E, Chouzenoux E, Pesquet JC. Stability of unfolded forward-backward to perturbations in observed data. In: Proc 31st Eur Signal Process Conf (EUSIPCO); 2023. p. 865--869.
\bibitem{fresnel_db}
Belkebir K, Saillard M. Special section: Testing inversion algorithms against experimental data. Inverse Probl. 2001;17(6):1565--1571; Geffrin JM, Sabouroux P, Eyraud C. Free space experimental scattering database continuation: experimental set-up and measurement precision. Inverse Probl. 2005;21(6):S117--S130.
\end{thebibliography}
"""

t2 = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", lambda m: new_bib.strip(), t, flags=re.S)
p.write_text(t2, encoding="utf-8")
print("done, old length:", len(t), "new:", len(t2))

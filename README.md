# Leuven.MapMatching

Align a trace of GPS measurements to a map or road segments.

The matching is based on a Hidden Markov Model (HMM) with non-emitting 
states. The model can deal with missing data and you can plug in custom
transition and emission probability distributions.

![example](http://people.cs.kuleuven.be/wannes.meert/leuvenmapmatching/example1.png?v=2)

Main reference:

> Meert Wannes, Mathias Verbeke, "HMM with Non-Emitting States for Map Matching",
> European Conference on Data Analysis (ECDA), Paderborn, Germany, 2018.

Other references:

> Devos Laurens, Vandebril Raf (supervisor), Meert Wannes (supervisor),
> "Trafﬁc patterns revealed through matrix functions and map matching",
> Master thesis, Faculty of Engineering Science, KU Leuven, 2018

## Installation and usage

    $ pip install leuvenmapmatching

More information and examples:

[leuvenmapmatching.readthedocs.io](https://leuvenmapmatching.readthedocs.io)

## Dependencies

Required:

- [numpy](http://www.numpy.org)
- [scipy](https://www.scipy.org)


Optional (only loaded when methods are called to rely on these packages):

- [matplotlib](http://matplotlib.org):
    For visualisation
- [smopy](https://github.com/rossant/smopy):
    For visualisation
- [nvector](https://github.com/pbrod/Nvector):
    For latitude-longitude computations
- [gpxpy](https://github.com/tkrajina/gpxpy):
    To import GPX files
- [pykalman](https://pykalman.github.io):
    So smooth paths using a Kalman filter
- [pyproj](https://jswhit.github.io/pyproj/):
    To project latitude-longitude coordinates to an XY-plane
- [rtree](http://toblerity.org/rtree/):
    To quickly search locations


## Contact

Wannes Meert, DTAI, KU Leuven  
wannes.meert@cs.kuleuven.be  
https://dtai.cs.kuleuven.be

Mathias Verbeke, Sirris  
mathias.verbeke@sirris.be  
http://www.sirris.be/expertise/data-innovation

Developed with the support of [Elucidata.be](http://www.elucidata.be).


## License

Copyright 2015-2018, KU Leuven - DTAI Research Group, Sirris - Elucidata Group  
Apache License, Version 2.0.

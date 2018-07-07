# Leuven.MapMatching

Align a trace of GPS measurements to a map or road segments.

The matching is based on a Hidden Markov Model (HMM) with non-emitting 
states. The model can deal with missing data and you can plug in custom
transition and emission probability distributions.

![example](http://people.cs.kuleuven.be/wannes.meert/dtaimapmatching/example1.png)

Reference:

> Meert Wannes, Mathias Verbeke, "HMM with Non-Emitting States for Map Matching",
> European Conference on Data Analysis (ECDA), Paderborn, Germany, 2018.

## Installation and usage

    $ pip install leuvenmapmatching

More information and examples:

[leuvenmapmatching.readthedocs.io](https://leuvenmapmatching.readthedocs.io)

## Dependencies

- [nvector](https://github.com/pbrod/Nvector)
- [gpxpy](https://github.com/tkrajina/gpxpy)
- [pykalman](https://pykalman.github.io)
- [numpy](http://www.numpy.org)
- [scipy](https://www.scipy.org)
- [matplotlib](http://matplotlib.org)


## Contact

Wannes Meert, DTAI, KU Leuven  
wannes.meert@cs.kuleuven.be  
https://dtai.cs.kuleuven.be

Mathias Verbeke, Sirris  
mathias.verbeke@sirris.be  
http://www.sirris.be/expertise/data-innovation

Developed with the support of [Elucidata.be](http://www.elucidata.be).


## License

Copyright 2015-2018, KU Leuven - DTAI Research Group, Sirris  
Apache License, Version 2.0.
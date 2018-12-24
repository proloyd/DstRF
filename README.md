# DstRF   
The magnetoencephalography (MEG) response to continuous auditory stimuli, such as speech, is commonly described using a 
linear filter, the auditory temporal response function (TRF). Though components of the sensor level TRFs have been well 
characterized, the cortical distributions of  the underlying neural responses are not well-understood. In our recent 
work, we provide a unified framework for determining the TRFs of neural sources directly from the MEG data, by 
integrating the TRF and distributed forward  source models into one, and casting the joint estimation task as a 
Bayesian optimization problem. Though the resulting  problem emerges as non-convex, we propose efficient solutions 
that leverage recent advances in evidence maximization. For more details please refer to the following resources:

1. P. Das, C. Brodbeck, J. Z. Simon, B. Babadi, [Direct Cortical Localization of the MEG Auditory Temporal Response 
Function: a Non-Convex Optimization Approach](https://isr.umd.edu/Labs/CSSL/simonlab/pubs/SFN2018.pdf); Proceedings 
of the 47th Annual Neuroscience Meeting (SfN 2018), Nov. 2-7, San Diego, CA.
2. P. Das, C. Brodbeck, J. Z. Simon, B. Babadi, [Cortical Localization of the Auditory Temporal Response Function from 
MEG via Non-Convex Optimization](https://isr.umd.edu/Labs/CSSL/simonlab/pubs/Asilomar2018.pdf); 2018 Asilomar Conference
 on Signals, Systems, and Computers, Oct. 28–31, Pacific Grove, CA (invited)
 
 This repository contains the implementation of our direct TRF estimation algorithm in python (version 3.6 and above). 
  
 
 Requirements:
 -----------
Eelbrain ([Download/ Installation Instructions](https://eelbrain.readthedocs.io/en/latest/reference.html))
 
 How to use:
 ----------
run
```python
h, model = dstrf(meg, stim, lead_field, noise, mu='auto', tstop=1.0, nlevels=2, n_splits=3, normalize='l1')
```
to perform a 3-fold cross-validation and then construct the model for 1s long TRF with the regularization 
weight among the given range that gives least generalization error. For more options, please look at the 
docstring.

Results
-------
We applied the algorithm on a subset of MEG data collected from 17 adults (aged 18-27 years) under an auditory task 
described in the papers. In short, during the task, the participants listened to `1 min` long segments from 
an audio-book recording of [The Legend of Sleepy Hollow by Washington Irving](https://librivox.org/the-legend-of-sleepy-hollow-by-washington-irving/), narrated by a male speaker. We consider localizing the TRFs using a total of `6 min` data from each participant. 
MNE-python 0.14 was used in pre-processing the raw data to automatically detect and discard flat channels, remove 
extraneous artifacts, and to band-pass filter the data in the range `1 - 80 Hz`. The six `1 min` long 
data epochs were then down-sampled to ``200 Hz``. As the stimulus variable, we used the speech envelope reflecting 
the momentary acoustic power, by averaging the auditory spectrogram representation (generated using a model of the 
auditory periphery) across the frequency bands, sampled at `200 Hz`.  A volume source space for individual subjects was 
defined on a 3D regular grid with a resolution of `7 mm` in each direction. The lead-field matrix was then computed by 
placing free orientation virtual dipoles on the resulting `3322` grid points. The consistent components of our estimated 
`1 s`-long 3D TRFs accross all `17` subjects looks like following:
 
 ![Demo](https://user-images.githubusercontent.com/28169943/49410670-bf51c500-f733-11e8-9894-43880aa8d49e.gif)
 
 Isn't that cool? Do expect to see something like that with any other source localization method? If you realize you could 
 use this method on your data, please feel free to use the codes. You can reach me at proloy@umd.edu if you have any 
 issues with the codes. And don't forget to go over the papers/ posters before applying the algorithm. 
 
 Note that, 
 this is a dev version, and I will be adding more functionality over time, so feel free to ask me 
 to add any other functionality, or report if anything si broken.
    
 Citation
 --------
 This repo is open for anyone to use under Apache license. But if you use this code for your publication, I will
 appreciate you if cite my papers mentioned above.
  
 

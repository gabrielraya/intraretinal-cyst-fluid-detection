# Intraretinal cyst fluid detection using Deep Learning


This repository contains the code for the development of a Intraretinal cyst fluild detection system, final project for the course on [Intelligent Systems in Medical Imaging 2018-2019](https://ismi19.grand-challenge.org/) at Radboud University, Nijmegen, The Netherlands.

![Retina](https://github.com/gabrielraya/intraretinal-cyst-fluid-detection/blob/master/images/project.png)



Outline: 

- [Intro](#Intro)
- [Related Work](#related-work)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Paper](#paper)
- [Meetings](#meetings)
- [Project Plan](#project-plan)
- [Important facts](#important-facts)




##  Intro
This work aims to do an experimental analysis of recent state of the art deep learning algorithms to  to solve the task of **Intraretinal cyst fuild detection**, specially to find three retinal fluid types, with annotated images provided by two clinical centers, which were acquired with the three most common [OCT](https://en.wikipedia.org/wiki/Optical_coherence_tomography) device vendors from
patients with two different retinal diseases.  



## Related Work

**Please read the following papers** (this section will be based on these papers and other related useful work we find):

- [Deep learning approach for the detection and quantification of intraretinal cystoid fluid in multivendor optical coherence tomography](https://www.osapublishing.org/boe/abstract.cfm?uri=boe-9-4-1545)
- [**RETOUCH** - The Retinal OCT Fluid Detection and
Segmentation Benchmark and Challenge](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8653407):  It covers much more than just cysts but it gives a good overview of different techniques.

***Please feel free to add references of other related useful work!!!***

***Remember:*** 
>  "***If I have seen further it is by standing on the shoulders of Giants***." *Isaac Newton*

## Data
The training data together with the annotations for the ICON challenge can be downloaded using the following link (*Please, make sure that you all read the data usage policy document included and understand what is stated there.*):


- [https://drive.google.com/open?id=1Y2XKig36mh2WuJ-1ArJWoF-tlgKp8UnT](https://drive.google.com/open?id=1Y2XKig36mh2WuJ-1ArJWoF-tlgKp8UnT)  
 
For a detailed explanation explanation of the data go to the [**RETOUCH Challenge**](https://retouch.grand-challenge.org/Details/), the `details` section explained it really good. 

If you have any questions, please contact Clarisa, the **supervisor** at [**Clara.SanchezGutierrez@radboudumc.nl**](Clara.SanchezGutierrez@radboudumc.nl)


## Methods
*In progress*

## Results
*This section will be updated soon, future work to be done is to evaluated the enhance and evaluate the algoritm in the following databases ICON challenge.*

The proposed method was evaluated in the [ICON challenge](https://icon.grand-challenge.org/
) database.


## Paper
To prepare our work as a Full Paper we used the LaTeX style files provided at:
[https://github.com/MIDL-Conference/MIDLLatexTemplate](https://github.com/MIDL-Conference/MIDLLatexTemplate), Latex template for the [MIDL Conference](https://midl.io/). Simple use of the JMLR / PMLR style.

You can find the overleaf link here:

[**https://www.overleaf.com/4136926763qjkzvpfyffbf**](https://www.overleaf.com/4136926763qjkzvpfyffbf)

## Meetings

### Minutes Meeting 4/24/19

- **Started at**: 12:00 hrs
- **Ended at**: ~13:00 hrs.
- **Brieft**: Brieft discussion about the logistics and goals of the project.


## Project Plan
|#     |  To do   |
|---------|-----------------|
|1 | Read the 2 papers    |
|2 | Understand the problem     |
|3 | Start with Introduction of the paper along with the abstract    |
|4 | Start a power point presentation for the midterm  |
|5 | Set deadlines    |
|6 | Check the data  |
|8 | Set a schedule to fins power workout days|
|9 | First team meeting intensive power workout day|
|11 | Maybe a brieft recap about CNN and Unet to put all team members in the same vibe |
|10 | Try to run a first CNN attempt |
|11 | Copy & paste the Unet from the assigment   |

### Team meetings
Please fill this form to set meeting days and time https://docs.google.com/spreadsheets/d/1KN2foBNWBkC64TE_TvXrZgBjbJK98aN7AKZDqhmMKkg/edit?usp=sharing


## Important facts

### Grading
- **Final Grade = 0.2 * Assignments + 0.6 * Final project + 0.2 * Exam**
- **Final project**:
The average of several grades, consisting in the final evaluation of the **final project**, of the **technical report**, of the peer evaluation, and of the answers to questions during the final presentation.


### Deadlines
- Friday 17/05 : **Midterm Presentation** |  9:00-12:00 -> Mercator 1, room 00.20
- Friday 21/06 : **Final Presentation**   |  9:00-12:00 -> Linnaeusbuilding, room 5 (LIN5), Heyendaalseweg 137
- Thursday 13/06: **Final Exam ISMI-19**


#### Team Members:

|Name     |  Slack Handle   |
|---------|-----------------|
|[Gabriel Raya Rodriguez](https://github.com/gabrielraya) |     @gabrielraya    |
|[Julius Mannes](https://github.com/JuliusMannes)| @JuliusMannes        |
|[Tristan Payer](https://github.com/sirtris) |     @sirtris    |
|[Pedro Martínez](https://github.com/PMedel) |     @PMedel    |






## Bibliography

[1] Freerk G. Venhuizen, Bram van Ginneken, Bart Liefers, Freekje van Asten, Vivian Schreur, Sascha Fauser, Carel Hoyng, Thomas Theelen, and Clara I. Sánchez, "Deep learning approach for the detection and quantification of intraretinal cystoid fluid in multivendor optical coherence tomography," Biomed. Opt. Express 9, 1545-1569 (2018)

[2] H. Bogunović et al., "RETOUCH -The Retinal OCT Fluid Detection and Segmentation Benchmark and Challenge," in IEEE Transactions on Medical Imaging.
doi: 10.1109/TMI.2019.2901398
keywords: {Retina;Image segmentation;Diseases;Biomedical imaging;Image analysis;Fluids;Benchmark testing;Evaluation;Image segmentation;Image classification;Optical Coherence Tomography;Retina},
URL: [http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8653407&isnumber=4359023](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8653407&isnumber=4359023)



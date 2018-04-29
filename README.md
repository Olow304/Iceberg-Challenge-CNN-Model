# Iceberg-Challenge-CNN-Model
let's predict whether an image is "iceberg" or "ship"

## Iceberg Classifier Challenge
Let's predict whether an image is a ship or an iceberg.

## About the Data
The remote sensing systems used to detect icebergs are housed on satellites over 600 kilometers above the Earth. The Sentinel-1 satellite constellation is used to monitor Land and Ocean. Orbiting 14 times a day, the satellite captures images of the Earth's surface at a given location, at a given instant in time. The C-Band radar operates at a frequency that "sees" through darkness, rain, cloud and even fog. Since it emits it's own energy source it can capture images day or night.

Satellite radar works in much the same way as blips on a ship or aircraft radar. It bounces a signal off an object and records the echo, then that data is translated into an image. An object will appear as a bright spot because it reflects more radar energy than its surroundings, but strong echoes can come from anything solid - land, islands, sea ice, as well as icebergs and ships. The energy reflected back to the radar is referred to as backscatter.


The data has two channels: HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). This can play an important role in the object characteristics, since objects tend to reflect energy differently. Easy classification examples are see below. These objects can be visually classified. But in an image with hundreds of objects, is very hard to classified.

###  Easy to classify
<img src="https://storage.googleapis.com/kaggle-media/competitions/statoil/8ZkRcp4.png" />

###  More challenging objects
Here we see challenging objects to classify. Is it a Ship or is it an Iceberg? 
<img src="https://storage.googleapis.com/kaggle-media/competitions/statoil/AR4NDrK.png" />


# Iceberg-Challenge-CNN-Model
let's predict whether an image is "iceberg" or "ship"

## Iceberg Classifier Challenge
Let's predict whether an image is a ship or an iceberg.

## About the Data
The remote sensing systems used to detect icebergs are housed on satellites over 600 kilometers above the Earth. <b>The Sentinel-1 satellite constellation is used to monitor Land and Ocean</b>. Orbiting 14 times a day, <b>the satellite captures images of the Earth's surface at a given location</b>, at a given instant in time. The C-Band radar operates at a frequency that "sees" through darkness, rain, cloud and even fog. Since it emits it's own energy source it can capture images day or night.

<b>Satellite radar works in much the same way as blips on a ship or aircraft radar. It bounces a signal off an object and records the echo, then that data is translated into an image</b>. An object will appear as a bright spot because it reflects more radar energy than its surroundings, but strong echoes can come from anything solid - land, islands, sea ice, as well as icebergs and ships. The energy reflected back to the radar is referred to as backscatter.

<img style="height:300px; width: 800px" src="https://storage.googleapis.com/kaggle-media/competitions/statoil/NM5Eg0Q.png" />

<b>The data has two channels:</b> <b>HH</b> (transmit/receive horizontally) and <b>HV</b> (transmit horizontally and receive vertically). This can play an important role in the object characteristics, since objects tend to reflect energy differently. Easy classification examples are see below. These objects can be visually classified. But in an image with hundreds of objects, is very hard to classified. 

#### Easy to classify
<img style="height:200px; width: 700px" src="https://storage.googleapis.com/kaggle-media/competitions/statoil/8ZkRcp4.png" />

<img style="height:200px; width: 700px" src="https://storage.googleapis.com/kaggle-media/competitions/statoil/M8OP2F2.png" />

#### More challenging objects
Here we see challenging objects to classify.  Is it a Ship or is it an Iceberg? 
<img style="height:200px; width: 700px" src="https://storage.googleapis.com/kaggle-media/competitions/statoil/AR4NDrK.png" />

<img style="height:200px; width: 700px" src="https://storage.googleapis.com/kaggle-media/competitions/statoil/nXK6Vdl.png" />

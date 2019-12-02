# Data and Motivation
Data comes from several field campaigns in which an aircraft mounted cloud particle imager (CPI)
took photos of ice crystals during flight. The CPIVIEW software was used
to cut and classify the data. CPIVIEW classification is user specified and usually very poor
as compared to manual classification, so it makes sense to build a convolutional neural network
to classify these particles by their shape. This is important because the shape determines
the scattering properties of the ice crystals, and given a distribution, of the clouds, and
ultamitely the Earth's atmosphere. Clouds are the most significant source of uncertainty in climate
models for this reason. Finding a fast and accurate classifier can help close this gap in understanding
so we can more accuratley predict the climate.

The data was processed at UIUC, choosen and manually classfied. Dr. Jun Umshik (former employee of UIUC DAS) 
put together this data, which is NOT available here. Contact me if you wish to know more. 

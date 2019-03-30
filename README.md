# optimal_ats_stock
Client Solution for finding Optimal ATS to route Portfolio of stocks.

Python Script with GUI


![alt text](https://github.com/sunsuntianyi/optimal_ats_stock/blob/master/demo.png)


Deliverable Tool User Manual
(File Name: “run.py”)


Python Module used: 

PySimpleGUI, pandas, functools, matplotlib, math, pylab

Before Starting up the tool:
Changing Dataset directory/Using New Dataset:
Open the “run.py” python script, scroll to the bottom where you see this:
Change the directory of the three datasets (line 442 - 444) to your dataset directory. For using new datasets, please be remind that the structure (variable name) of the datasets have to be the same as the original datasets. 


Starting up the tool:

	Option 1: Run in Terminal
		change directory to the python file (deliverable tool) folder
		“python run.py”
	Option 2: Run in Python Shell/Python IDE
		execute the ‘run.py’ file in shell/IDE

Interface:












Portfolio: This is the portfolio of stocks the user wants to trade. Enter as many stocks as you want. Separate each stock using comma (‘,’). space between comma as well as small/capital letter will not impact the result.

Share: The share for each individual stock you previously entered above. The order of input should be the same as the order of the stock entered in the portfolio. Separate each stock using comma (‘,’). space between comma as well as small/capital letter will not impact the result.

Data Timeline: This is a dropdown which contains three options: 
	“Most Recent Week” - Using most recent week of the data
	“Most Recent Month” - Using most recent month of the data
	“All Data” - Using the whole dataset

Number of Radar Chart: This is the number of radar chart will be shown in the output.
Indices Weight: This is input weight for each of the five calculated indices. User decide how much weight to put on each index. The overall rank is calculated based on the weighted value of these five index. Default is 1. Refer to project paper for the detail of how it is calculated.

	SPI: Share Proportion Index
	TPI: Trade Proportion Index
	SLI: Sector Liquidity Index
	PRI: Participation Rate Index
	ATSI: Average Trade Size Index
There will be two output windows. The radar chart on the bottom left shows top three ATS which the rank based on the weighted value of these five index. The smaller window on the bottom right shows top six ATS. The bottom two lines shows the percentage of the particular stock is traded on ATS. 

Analyzing another portfolio:
Click “OK” to close the bottom right window. Then input your portfolio information. (User doesn’t have to close the radar chart window as it will automatically updated.)

Exiting the Tool:
Click “OK” to close the bottom right window. Then click “Exit” in the main program window.

var TDSCanvas = document.getElementById("totaldatasize");

Chart.defaults.global.defaultFontFamily = "Lato";
Chart.defaults.global.defaultFontSize = 14;

var dcc_rcs = document.getElementById("dcc_rcs").value.split(',');
var dcc_cur = document.getElementById("dcc_cur").value.split(',');
var dcc_pre = document.getElementById("dcc_pre").value.split(',');
var dcc_mds = document.getElementById("dcc_mds").value;

var TDSData = {
	labels: dcc_rcs,
	datasets: [{
		label: "previous round",
		backgroundColor: "transparent",
		borderColor: "rgba(200,0,0,0.6)",
		fill: false,
		radius: 6,
		pointRadius: 6,
		pointBorderWidth: 3,
		pointBackgroudColor: "orange",
		pointBorderColor: "rgba(200,0,0,0.6)",
		pointHoverRadius: 10,
		data: dcc_pre
	}, {
		label: "current round",
		backgroundColor: "transparent",
		borderColor: "rgba(0,0,200,0.6)",
		fill: false,
		radius: 6,
		pointRadius: 6,
		pointBorderWidth: 3,
		pointBackgroundColor: "cornflowerblue",
		pointBorderColor: "rgba(0,0,200,0.6)",
		pointHoverRadius: 10,
		data: dcc_cur
	}]
};

var chartOptions = {
	scale: {
		gridLines: {
			color: "black",
			lineWidth: 2
		},
		angleLines: {
			display: false			
		},
		ticks: {
			display: false,
			beginAtZero: true,
			min: 0,
			max: parseInt(dcc_mds), // max is from datasize
			stepSize: 2000
		},
		pointLabels: {
			fontSize: 14,
			fontColor: "black"
		}
	},
	legend: {
		position: 'top',
		labels: {
			font: {
				size: 14	
			}
		}
	}
};

var radarChart = new Chart(TDSCanvas, {
	type: 'radar',
	data: TDSData,
	options: chartOptions
});

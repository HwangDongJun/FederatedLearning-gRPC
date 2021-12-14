const onvif = require("node-onvif");

// found onvif camera link
//console.log('Start the discovery process.');
//onvif.startProbe().then((device_info_list) => {
//	console.log(device_info_list.length + ' devices were found.');
//	device_info_list.forEach((info) => {
//		console.log('- ' + info.urn);
//		console.log('  - ' + info.name);
//		console.log('  - ' + info.xaddrs[0]);
//	});
//}).catch((error) => {
//	console.error(error);
//});

/*
let device = new onvif.OnvifDevice({
	xaddr: "http://192.168.1.150:2020/onvif/device_service",
	user : "dnlab2021",
	pass : "dnlab2021"
});

device.init().then(() => {
		let url = device.getUdpStreamUrl();
		console.log(url);
	}).catch((error) => {
		console.error(error);
	});
*/

Stream = require('node-rtsp-stream');
stream = new Stream({
	name: '0',
	streamUrl: 'rtsp://192.168.1.150:554/stream1',
	wsPort: 9999
});	

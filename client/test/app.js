const express = require('express');
const app = express();
const path = require('path');

app.set('view engine', 'jade');
app.set('views', './views');

app.use(express.static(path.join(__dirname, '/')));

app.get('/', (req, res) => {
	res.redirect('/FLclient1');
});

app.listen(5007);

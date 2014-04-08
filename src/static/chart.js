$(document).ready(
    
    function() {
      
      Highcharts.setOptions({
        global: {
            timezoneOffset: -4 * 60
        }
      });
      
      $('#container')
          .highcharts(
              {
                chart : {
                  type : 'spline'
                },
                title : {
                  text : 'Показания счетчика газа'
                },
                xAxis : {
                  type : 'datetime',
                  dateTimeLabelFormats : { // don't display the dummy year
                    month : '%e. %b',
                    year : '%b'
                  }
                },
                yAxis : {
                  title : {
                    text : 'Расход (куб.м./сутки)'
                  },
                  plotLines: [{
                    color: 'red',
                    value: average_consumption,
                    width: 2,
                    label: {
                      text: '<b>'+average_consumption+'</b> - Средний расход за период (куб.м./сутки)',
                      align: 'left',
                      y: 16
                    }
                  }]
                },
                tooltip : {
                  formatter: function() {
                    return Highcharts.dateFormat('%e %b %H:%M', this.x) + '<br>Расход: <b>' + this.y + '</b> куб.м./сутки<br>Показания счетчика: '+ this.point.options.value + '<br>';
                  }
                },
                series : [ {
                  name : 'Расход газа',
                  // Define the data points. All series have a dummy year
                  // of 1970/71 in order to be compared on the same x axis. Note
                  // that in JavaScript, months start at 0 for January, 1 for
                  // February etc.
                  data : series
                }]});
    });
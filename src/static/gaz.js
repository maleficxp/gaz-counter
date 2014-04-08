$(document).ready(function(){
  $(".digit-result").focus(function (){
    $(this).select();
  }).mouseup(function(e){
    e.preventDefault();
  }).keyup(function(){
    getNext(this).focus();
  });    
  
  $(".datepicker").datepicker($.datepicker.regional["ru"]);
});

function getNext(input) {
  var td = $(input).closest("td");
  var next_td = td.next("td.digit-cell");
  if (next_td.size()==0) {
    var tr = $(td.closest("tr"));
    var next_tr = tr.next("tr");
    next_td = next_tr.find("td.digit-cell:first");
  }
  var next_input = next_td.find(".digit-result");
  if (next_input.size()>0 && next_input.val()!="?")
    return getNext(next_input);
  else
    return next_input;
}
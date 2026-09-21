(function(){
  var m=/[?&]region=([0-9a-fA-F]+)-([0-9a-fA-F]+)/.exec(location.search);
  if(!m) return;
  var start=parseInt(m[1],16), end=parseInt(m[2],16);
  var lo=Math.min(start,end), hi=Math.max(start,end);
  var rows=document.querySelectorAll("tr.asm[id^='pc-']");
  for(var i=0;i<rows.length;i++){
    var pc=parseInt(rows[i].id.slice(3),16);
    if(pc===start||pc===end) rows[i].classList.add("region-edge");
    else if(pc>lo && pc<hi) rows[i].classList.add("region-band");
  }
  var head=document.getElementById("pc-"+start.toString(16));
  if(head) head.scrollIntoView({block:"center"});
})();

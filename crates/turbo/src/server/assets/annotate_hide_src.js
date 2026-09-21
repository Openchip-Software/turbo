(function(){
  var b=document.getElementById("toggle-src"); if(!b) return;
  function sync(){ b.textContent=document.documentElement.classList.contains("hide-src")?"Show source":"Hide source"; }
  function toggle(){
    var on=document.documentElement.classList.toggle("hide-src");
    localStorage.setItem("hidesrc",on?"1":"0"); sync();
  }
  sync();
  b.addEventListener("click",toggle);
  document.addEventListener("keydown",function(e){
    if(e.key!=="s"&&e.key!=="S") return;
    if(e.ctrlKey||e.metaKey||e.altKey) return;
    var t=e.target, tag=t&&t.tagName;
    if(tag==="INPUT"||tag==="TEXTAREA"||(t&&t.isContentEditable)) return;
    e.preventDefault(); toggle();
  });
})();

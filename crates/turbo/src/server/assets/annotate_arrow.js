(function(){
  var NS="http://www.w3.org/2000/svg", svg=null;
  function clear(){ if(svg) svg.innerHTML=""; var p=document.querySelector("tr.arrow-hl"); if(p) p.classList.remove("arrow-hl"); }
  function draw(a){
    var dst=document.getElementById(a.getAttribute("href").slice(1)); if(!dst) return;
    if(!svg){ svg=document.createElementNS(NS,"svg"); svg.id="br-arrow"; document.body.appendChild(svg); }
    clear(); dst.classList.add("arrow-hl");
    var ins=dst.querySelector("td.ins")||dst;
    var r1=a.getBoundingClientRect(), r2=ins.getBoundingClientRect();
    // The instruction column is the rightmost, width:100% column, so its text
    // is left-aligned at r2.left -- aim the arrowhead there (start of the
    // disassembly text), not r2.right (the far page edge).
    var x1=r1.right, y1=r1.top+r1.height/2, x2=r2.left, y2=r2.top+r2.height/2;
    // Smooth S-connector: both control points give a horizontal *forward*
    // (rightward) tangent -- the curve leaves the right of the [branch] box
    // heading toward the code, and arrives into the instruction from the left
    // (so the arrowhead points right, into the text). No backward "left then
    // right" kink. The horizontal bulge scales with the vertical distance so
    // short hops curve gently and long jumps arc wider.
    var dx=Math.max(24, Math.min(90, Math.abs(y2-y1)*0.5));
    var line=document.createElementNS(NS,"path");
    line.setAttribute("d","M"+x1+" "+y1+" C "+(x1+dx)+" "+y1+","+(x2-dx)+" "+y2+","+x2+" "+y2);
    line.setAttribute("class","br-line"); svg.appendChild(line);
    var head=document.createElementNS(NS,"path");
    head.setAttribute("d","M"+x2+" "+y2+" l -6 -4 m 6 4 l -6 4"); // tip points right, into the instruction
    head.setAttribute("class","br-line"); svg.appendChild(head);
  }
  document.addEventListener("mouseover",function(e){ var a=e.target.closest&&e.target.closest("a.brj"); if(a) draw(a); });
  document.addEventListener("mouseout",function(e){ if(e.target.closest&&e.target.closest("a.brj")) clear(); });
})();

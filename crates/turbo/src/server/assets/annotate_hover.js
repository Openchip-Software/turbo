(function(){
  var panel=document.getElementById("vhover"); if(!panel) return;
  var body=panel.querySelector(".vhover-body");
  function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
  function fill(tr){
    var insEl=tr.querySelector("td.ins"), pcEl=tr.querySelector("td.pc");
    var vmin=tr.getAttribute("data-vlmin"), vmax=tr.getAttribute("data-vlmax");
    var vl = vmin===vmax ? vmin : vmin+"–"+vmax;
    var ins=(pcEl?pcEl.textContent+"  ":"")+(insEl?insEl.textContent:"");

    // Find the nearest VL histogram at or above this row, in the same table.
    var hist=null, srcPc=null, isSelf=false;
    var tbl=tr.closest("table");
    if(tbl){
      var rows=tbl.querySelectorAll("tr.asm"), idx=-1;
      for(var i=0;i<rows.length;i++){ if(rows[i]===tr){ idx=i; break; } }
      for(var j=idx;j>=0;j--){
        var h=rows[j].querySelector(".vec .vhist");
        if(h){ hist=h; var p=rows[j].querySelector("td.pc"); srcPc=p?p.textContent:null; isSelf=(j===idx); break; }
      }
    }

    var histHtml="";
    if(hist){
      var label=isSelf?"VL histogram (this vsetvl)":("VL histogram &middot; set at "+esc(srcPc||"?"));
      histHtml="<div class=\"vhover-sub\">"+label+"</div>"+hist.outerHTML;
    } else {
      histHtml="<div class=\"vhover-sub\">no vsetvl histogram above</div>";
    }

    // The per-op grid only applies to actual vector ops (data-v* present);
    // for a vsetvl* row the histogram head already carries SEW/LMUL/elems.
    var grid = tr.hasAttribute("data-vsew")
      ? "<div class=\"vhover-grid\">"+
          "<span>SEW</span><b>"+esc(tr.getAttribute("data-vsew")||"")+"</b>"+
          "<span>LMUL</span><b>"+esc(tr.getAttribute("data-vlmul")||"—")+"</b>"+
          "<span>VL</span><b>"+esc(vl||"")+"</b>"+
          "<span>exec</span><b>"+esc(tr.getAttribute("data-vexec")||"")+"</b>"+
          "<span>elems</span><b>"+esc(tr.getAttribute("data-velems")||"")+"</b>"+
          cacheRows(tr)+
        "</div>"
      : "";

    // nosemgrep: insecure-document-method -- every interpolation goes through esc(); hist is our own DOM node
    body.innerHTML = "<div class=\"vhover-ins\">"+esc(ins)+"</div>"+grid+cacheBars(tr)+histHtml;
  }
  // Modelled cache behaviour, present only on instructions that touched memory
  // in a run processed with --cache-sim. Vector data-side only: no scalar
  // accesses and no instruction fetch reach the model.
  function cacheRows(tr){
    if(!tr.hasAttribute("data-l1h")) return "";
    var h=+tr.getAttribute("data-l1h"), m=+tr.getAttribute("data-l1m");
    var rate=(h+m)>0 ? (100*h/(h+m)).toFixed(1)+"%" : "—";
    return "<span>L1 h/m</span><b>"+esc(h+" / "+m)+"</b>"+
           "<span>L1 rate</span><b>"+esc(rate)+"</b>"+
           "<span>L2 h/m</span><b>"+
             esc(tr.getAttribute("data-l2h")+" / "+tr.getAttribute("data-l2m"))+"</b>";
  }
  // The same counts as a two-bar infographic: one bar per level, green from 0 to
  // hits and red from hits out to that level's accesses.
  //
  // Both bars share one x axis whose max is L1 accesses, because the L2 only
  // ever sees what missed the L1 -- so its bar's *length* is its share of all
  // traffic and the hit/miss split within it is its own rate. Normalising each
  // bar to its own total would draw a 64-lookup L2 as wide as a 2.7M-access L1
  // and lose exactly the thing worth seeing. A non-empty level always gets at
  // least a hairline so "tiny" stays distinguishable from "absent".
  function cacheBars(tr){
    if(!tr.hasAttribute("data-l1h")) return "";
    var l1h=+tr.getAttribute("data-l1h"), l1m=+tr.getAttribute("data-l1m");
    var l2h=+tr.getAttribute("data-l2h")||0, l2m=+tr.getAttribute("data-l2m")||0;
    var axis=l1h+l1m;
    if(axis<=0) return "";
    function fmt(n){ return n.toLocaleString("en-US"); }
    function bar(name,h,m){
      var tot=h+m;
      // Widths are percentages of the axis; the green/red pair together span the
      // level's own share of it, leaving the remainder as empty track.
      var pct=function(n){ return tot>0 ? Math.max(0.6, 100*n/axis) : 0; };
      var rate=tot>0 ? (100*h/tot).toFixed(1)+"%" : "—";
      return "<div class=\"cbar-row\">"+
        "<span class=\"cbar-lbl\">"+name+"</span>"+
        "<span class=\"cbar-track\">"+
          (tot>0
            ? "<i class=\"cbar-hit\" style=\"width:"+pct(h).toFixed(3)+"%\"></i>"+
              "<i class=\"cbar-miss\" style=\"width:"+pct(m).toFixed(3)+"%\"></i>"
            : "")+
        "</span>"+
        "<span class=\"cbar-rate\">"+rate+"</span>"+
        "</div>";
    }
    return "<div class=\"vhover-sub\">cache &middot; "+fmt(axis)+" accesses</div>"+
      "<div class=\"cbar\">"+
        bar("L1",l1h,l1m)+
        bar("L2",l2h,l2m)+
        "<div class=\"cbar-key\">"+
          "<i class=\"cbar-hit\"></i>hit<i class=\"cbar-miss\"></i>miss"+
        "</div>"+
      "</div>";
  }
  document.addEventListener("mouseover",function(e){
    var tr=e.target.closest&&e.target.closest("tr.asm");
    if(!tr) return;
    // Trigger for vector ops and for vsetvl* rows that carry a histogram.
    if(tr.hasAttribute("data-vsew") || tr.querySelector(".vec .vhist")) fill(tr);
  });
})();

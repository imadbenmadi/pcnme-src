"""Check that the PCNME default numbers (final doc, section 18; config/default.yaml) fit together.

Run: python3 calc_defaults.py. Placeholders (M) must be replaced by measured values, then rerun.
"""
import math
ms=1e-3; KB=1e3*8  # bits
# task types: share, size_bits, D_loose, D_tight, class, GPU mean run on fog (s), CPU mean (s), cloud GPU (s)
T={
 'T1':dict(share=0.73,S=150*KB,D=(0.100,0.050),cls='hard',gpu=8*ms, cpu=90*ms, cl=4*ms, Sr=3*KB),
 'T3':dict(share=0.15,S=150*KB,D=(0.100,0.070),cls='hard',gpu=10*ms,cpu=110*ms,cl=5*ms, Sr=3*KB),
 'T4':dict(share=0.10,S=150*KB,D=(1.0,0.5),  cls='soft',gpu=25*ms,cpu=200*ms,cl=12*ms,Sr=3*KB),
 'T2':dict(share=0.02,S=60*150*KB,D=(3.0,1.1*0.720),cls='soft',gpu=720*ms,cpu=None,cl=200*ms,Sr=10*KB),
}
assert abs(sum(t['share'] for t in T.values())-1)<1e-9
c_gpu=2; zones=4
Emix=sum(t['share']*t['gpu'] for t in T.values())
cap=c_gpu/Emix
print(f"mean GPU work per task {Emix*1e3:.1f} ms; zone GPU capacity {cap:.1f} tasks/s")
alpha=0.8
for name,t in T.items():
    for i,Dv in enumerate(t['D']):
        b = t['gpu']>=alpha*Dv
        print(f"  {name} D={Dv}s {'boulder' if b else 'pebble'} (Ê_strong={t['gpu']}s vs αD={alpha*Dv:.3f})")
mean_bits=sum(t['share']*t['S'] for t in T.values())
soft_bits=sum(t['share']*t['S'] for t in T.values() if t['cls']=='soft')
print(f"mean input {mean_bits/8e3:.0f} KB; cloud-eligible (soft) bits per task {soft_bits/8e3:.0f} KB")
for S,rho,ti in [('S1',0.5,0.5),('S2',0.9,0.9),('S3',1.5,1.5)]:
    lam_zone=rho*cap; lam_city=lam_zone*zones
    B_wan=lam_city*soft_bits/ti
    for cars in (200,1000):
        print(f"{S}: rho_fog {rho} -> {lam_zone:.1f} tasks/s/zone; per car {lam_zone/(cars/zones):.3f}/s at {cars} cars; B_WAN {B_wan/1e6:.0f} Mbps for TI_WAN {ti}")
# radio TI per zone at S2 with 600 Mbps typical Friis rate
lam=0.9*cap; print(f"radio TI per zone at S2 (600 Mbps): {lam*mean_bits/600e6:.2f}")
# energy
PUE_fog,PUE_cloud=1.5,1.1; p_gpu,p_cpu,p_cl=25,20,150
e_bh=0.14e-9*1.5; e_wan=0.23e-9*1.5; e_down=4.2e-9
for name,t in T.items():
    Eret=(2*e_bh+e_down)*t['Sr']
    Eref=PUE_fog*p_gpu*t['gpu']+Eret
    # worst option with p95 = 1.3 x mean (placeholder)
    opts={'fog_gpu':PUE_fog*p_gpu*1.3*t['gpu']+e_bh*t['S']+Eret}
    if t['cpu']: opts['fog_cpu']=PUE_fog*p_cpu*1.3*t['cpu']+e_bh*t['S']+Eret
    opts['cloud']=PUE_cloud*p_cl*1.3*t['cl']+(e_bh+e_wan)*t['S']+(e_wan+e_bh+e_down)*t['Sr']
    print(name,f"E_ref {Eref:.3f} J;", {k:round(v/Eref,2) for k,v in opts.items()}, f"net share of E_ref for a neighbour: {e_bh*t['S']/Eref:.4f}")
# cloud latency feasibility for T1 (mean delays)
d_bh=3*ms; d_wan=62.5*ms; B=600e6
Tin=d_bh+T['T1']['S']/B+d_wan; Tret=d_wan+d_bh+3*ms
print(f"T1 cloud mean: in {Tin*1e3:.0f} + run {T['T1']['cl']*1e3:.0f} + ret {Tret*1e3:.0f} = {(Tin+T['T1']['cl']+Tret)*1e3:.0f} ms vs D 100 ms")
# NSGA-II window size
lam_city=0.9*cap*zones; Twin=2.0
print(f"S2 window {Twin}s -> {lam_city*Twin:.0f} genes; 100x100 evaluations")
# gamma horizon
dt=1/(0.9*cap); print(f"S2 Δt between decisions at an anchor {dt*1e3:.1f} ms; γ^(Δt/τ)={0.9**(dt/0.1):.4f}; effective horizon {1/(1-0.9**(dt/0.1)):.0f} decisions")
# queue sanity: hard wait if queue full
print(f"GPU queue Q=32 full of T1: wait {32*8/2:.0f} ms (> D, so 'late' binds before 'full' for hard tasks)")

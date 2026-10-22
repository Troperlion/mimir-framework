float norm_join(float a,float b){float hi=max(fabs(a),fabs(b)),lo=min(fabs(a),fabs(b));if(hi==0.0f)return 0.0f;float q=lo/hi;return hi*sqrt(1.0f+q*q);}
#define F(k) as_float(args[3+(k)])
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE)
__kernel void optimizer(__global float* memory,__global const int* args) {
 int i=get_global_id(0),lid=get_local_id(0),group=get_group_id(0);
 __local float sumsA[256],sumsB[256];

int n=args[1],kind=args[2],pass=args[0],groups=(n+255)/256;
int mo=args[20],vo=args[21],go=args[22],uo=args[23],ro=args[24],flags=args[25];
// Compact layout and active moments are supplied by the host; scratch stays on device.
if(kind==0) {
 if(pass==0 && i<n)memory[i]-=F(0)*memory[go+i];
 return;
}
if(pass==0 && i<n) {
 float w=memory[i],g=memory[go+i],m=(flags&1)!=0?memory[mo+i]:0.0f,v=(flags&2)!=0?memory[vo+i]:0.0f,u=g;
 float b1=F(1),b2=F(2),eps=F(3),wd=F(4),bc1=F(5),bc2=F(6);
 if(kind==1 || kind==2 || kind==5 || kind==6 || kind==8) {
  m=b1*m+(1.0f-b1)*g;v=b2*v+(1.0f-b2)*g*g;
  float mh=m/bc1,vh=v/bc2;
  u=mh/(sqrt(vh)+eps);
  if(kind==5)u=F(14)>5.0f?F(13)*mh/(sqrt(vh)+eps):mh;
  if(kind==6)u=(b1*mh+(1.0f-b1)*g/bc1)/(sqrt(vh)+eps);
  if(kind==8)u+=wd*w;
 } else if(kind==3) {
  float update=b1*m+(1.0f-b1)*g;
  u=(float)(update>0.0f)-(float)(update<0.0f)+wd*w;
  m=b2*m+(1.0f-b2)*g;
 } else if(kind==4) {
  v=F(7)*v+(1.0f-F(7))*g*g;u=g/sqrt(v+max(1e-30f,eps));
 } else if(kind==7) {
  v=F(15)*v+(1.0f-F(15))*g*g;u=g/(sqrt(v)+eps);
 }
 if((flags&1)!=0)memory[mo+i]=m;
 if((flags&2)!=0)memory[vo+i]=v;
 if(kind==4 || kind==8)memory[uo+i]=u;
 if(kind!=4 && kind!=8)memory[i]=kind==2?w-F(0)*u-F(0)*(F(4)*w):w-F(0)*u;
}
if(pass==1 || pass==3) {
 float a=0.0f,b=0.0f;
 if(pass==1){if(i<n){float u=memory[uo+i],w=memory[i];a=fabs(u)/sqrt((float)(n));b=fabs(w)/sqrt((float)(n));}}
 else {for(int j=lid;j<groups;j+=256){a=norm_join(a,memory[ro+2*j]);b=norm_join(b,memory[ro+2*j+1]);}}
 sumsA[lid]=a;sumsB[lid]=b;BARRIER;
 for(int stride=128;stride>0;stride/=2){if(lid<stride){sumsA[lid]=norm_join(sumsA[lid],sumsA[lid+stride]);sumsB[lid]=norm_join(sumsB[lid],sumsB[lid+stride]);}BARRIER;}
 if(lid==0){int at=ro+2*(pass==1?group:groups);memory[at]=sumsA[0];memory[at+1]=sumsB[0];}
}
if(pass==2 && i<n) {
 float w=memory[i],u=memory[uo+i],lr=F(0);
 if(kind==4) {
  float rms=memory[ro+2*groups];float clip=1.0f/max(1.0f,rms/F(9));
  float scale=F(12)>0.0f?max(F(10),memory[ro+2*groups+1]):1.0f;
  u=clip*memory[go+i]/sqrt(memory[vo+i]+max(1e-30f,F(3)));
  if(F(8)>0.0f){float m=F(8)*memory[mo+i]+(1.0f-F(8))*u;memory[mo+i]=m;u=m;}
  w-=F(11)*scale*u;
 } else if(kind==8) {
  float wn=memory[ro+2*groups+1]*sqrt((float)(n)),un=memory[ro+2*groups]*sqrt((float)(n));float ratio=wn>0.0f && un>0.0f?wn/un:1.0f;
  w-=lr*ratio*u;
 } else if(kind==2)w=w-lr*u-lr*(F(4)*w);
 else w-=lr*u;
 memory[i]=w;
}

}

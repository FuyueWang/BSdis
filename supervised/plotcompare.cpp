#include "../../txt/supervised/Fit_C1B_v23_latest_e16__DATA00_22__ite10__Eloop_5_rebin100_rebin100__zero_fit11110030000__fb+_vb03us_FBnull_cs137Br.h"
#include </home/ubuntu/fuyuew/mrpc/MRPCProject/plotfunctions.cpp>


void plotcompare(){
  TString thisdir="supervised/";
  TString txtdir="../../txt/"+thisdir;
  TString source="Cs137";


  //Yang LiTao data:
  const Int_t Nbin=230;
  TH1D *YinitEhist=new TH1D("Yenergy","",Nbin,0.16,11.66);
  Double_t Yrelativeerr[Nbin];
  for(Int_t bini=0;bini<Nbin;bini++){
	 YinitEhist->SetBinContent(bini+1,cpkkd_cs137br[bini][1]);
	 Yrelativeerr[bini]=cpkkd_cs137br[bini][2]/cpkkd_cs137br[bini][1];
	 // YenergyCs137[bini]=cpkkd_cs137br[bini][0];
	 // YcountCs137[bini]=cpkkd_cs137br[bini][1];
	 // YerryCs137[bini]=cpkkd_cs137br[bini][2];
	 // YerrxCs137[bini]=0;
  }
  YinitEhist->Scale(1./YinitEhist->Integral());
  TH1D* Ehist[2];
  Ehist[0]=new TH1D("YEhist","",Nbin,0.16,11.66);
  Ehist[1]=new TH1D("CNNEhist","",Nbin,0.16,11.66);
  for(Int_t bini=0;bini<Nbin;bini++){
  	 Ehist[0]->SetBinContent(bini+1,YinitEhist->GetBinContent(bini+1));
  	 Ehist[0]->SetBinError(bini+1,Yrelativeerr[bini]*YinitEhist->GetBinContent(bini+1));
  }
  Ehist[0]->Draw();


  //NN result
  ifstream infile(txtdir+"bulk.csv", ios::in);
  TString tmpstr;
  Double_t tmpdouble;
  infile>>tmpstr>>tmpstr;
  for(Int_t bini=0;bini<Nbin;bini++){
	 infile>>tmpdouble;
	 infile>>tmpdouble; Ehist[1]->SetBinContent(bini+1,tmpdouble);
	 infile>>tmpdouble;
  }
  Ehist[1]->Scale(1./Ehist[1]->Integral());

  plotpara p1;
  p1.canvassize[0]=1000;
  p1.xname="E [keVee]";
  p1.yname[0]="Counts [Normalized]";
  p1.textcontent="Energy of the Bulk Events: "+source;
  p1.legendname.push_back("Ratio");
  p1.legendname.push_back("CNN");

  DrawNHistogram(Ehist,2,p1);
  
  // TCanvas *c1 = new TCanvas("c1","Plot",0,0,p1.canvassize[0],p1.canvassize[1]);
  // c1->cd();
  // TPad *pad1 = new TPad("pad1","",0,0,1,1);
  // pad1->Draw();
  // pad1->SetLeftMargin(p1.leftmargin);
  // pad1->SetRightMargin(p1.rightmargin);
  // pad1->SetTopMargin(p1.topmargin);
  // pad1->SetBottomMargin(p1.bottommargin);
  // pad1->SetFillStyle(4000); 
  // pad1->SetFillColor(0);
  // pad1->SetFrameFillStyle(4000);
  // pad1->cd();
  
  // NNEhist->SetLineColor(2);
  // NNEhist->Draw("same");

}

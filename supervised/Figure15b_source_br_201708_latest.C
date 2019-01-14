#include "Fit_C1B_v23_latest_e16__DATA00_22__ite10__Eloop_5_rebin100_rebin100__zero_fit11110030000__fb+_vb03us_FBnull_cs137Br.h"
#include "Fit_C1B_v23_latest_e16__DATA00_22__ite10__Eloop_5_rebin100_rebin100__zero_fit11110030000__fb+_vb03us_FBnull_co60Br.h"
#include "ERR_68.h"

void Figure15b_source_br_201708_latest()
{
	gStyle->SetOptStat(0);

	TCanvas *c1 = new TCanvas("c1","c1",1200,800);	
	//gPad->SetLogy();

	const int Nbin=230;

        Float_t energy[Nbin],ee[Nbin];
        Float_t N1[Nbin];
        Float_t eN1_u[Nbin],eN1_d[Nbin];
        Float_t N2[Nbin];
        Float_t eN2_u[Nbin],eN2_d[Nbin];
	for(unsigned int nb=0;nb<Nbin;nb++)
        	{
                energy[nb] = cpkkd_cs137br[nb][0];
                ee[nb] = 0.0;

                N1[nb] = cpkkd_cs137br[nb][3];
                if(N1[nb]<11){
                       for(int jj=0;jj<11;jj++){
                            if(N1[nb] == poi_err[jj][0]) {
                                        eN1_d[nb] = poi_err[jj][0]-poi_err[jj][1];
                                        eN1_u[nb] = poi_err[jj][2]-poi_err[jj][0];
                                        break;
                                                }
                                        }
                                }
                else{
                        eN1_u[nb] = eN1_d[nb] = sqrt(N1[nb]);
                        }

                N2[nb] = cpkkd_co60br[nb][3];
                if(N2[nb]<11){
                       for(int jj=0;jj<11;jj++){
                            if(N2[nb] == poi_err[jj][0]) {
                                        eN2_d[nb] = poi_err[jj][0]-poi_err[jj][1];
                                        eN2_u[nb] = poi_err[jj][2]-poi_err[jj][0];
                                        break;
                                                }
                                        }
                                }
                else{
                        eN2_u[nb] = eN2_d[nb] = sqrt(N2[nb]);
                        }

		}


Float_t NBr1[Nbin];
Float_t euNBr1[Nbin],edNBr1[Nbin];
Float_t NBr2[Nbin];
Float_t euNBr2[Nbin],edNBr2[Nbin];

for(unsigned int nb=0;nb<Nbin;nb++)
{
  NBr1[nb] = cpkkd_cs137br[nb][1];
  euNBr1[nb] = cpkkd_cs137br[nb][2];
  edNBr1[nb] = cpkkd_cs137br[nb][2];

  NBr2[nb] = cpkkd_co60br[nb][1];
  euNBr2[nb] = cpkkd_co60br[nb][2];
  edNBr2[nb] = cpkkd_co60br[nb][2];
}

        // TFile *infile = new TFile("C1B_dead0.81_Cs_bin50ev_ebulk_top_new.root");
        // TH1F *hraw = (TH1F*)infile->Get("cs_bulk_sim2");

        // TFile *infile2 = new TFile("C1B_dead0.81_Co_bin50ev_ebulk_top_new.root");
        // TH1F *hraw2 = (TH1F*)infile2->Get("co_bulk_sim2");

        const int numb = 400;
        float xmin = 0;
        float xmax = 20;
        float bin = 0.05;

        Double_t E_st=3.0;
        Double_t E_ed=5.0;

	Double_t N_cs = 0;
        Double_t N_co = 0;
        // for(int i=(E_st-xmin)/bin;i<(E_ed-xmin)/bin;i++)
        // {
        //   N_cs = N_cs + (hraw->GetBinContent(i+1));
        //   N_co = N_co + (hraw2->GetBinContent(i+1));
        // }
        // cout<<"N_cs = "<<N_cs<<endl;
        // cout<<"N_co = "<<N_co<<endl;

        Double_t M_cs = 0;
        Double_t M_co = 0;
        for(int i=0;i<Nbin;i++)
        {
	if(energy[i]>E_st&&energy[i]<E_ed)
        	{
		M_cs = M_cs + NBr1[i];
          	M_co = M_co + NBr2[i];
		}
        }
        cout<<"M_cs = "<<M_cs<<endl;
        cout<<"M_co = "<<M_co<<endl;

        Double_t scale_cs = (Double_t)(M_cs/N_cs);
        cout<<"scale_cs = "<<scale_cs<<endl;
        Double_t scale_co = (Double_t)(M_co/N_co);
        cout<<"scale_co = "<<scale_co<<endl;

	Double_t k=100.;

	// hraw->Scale(scale_cs);
	// hraw2->Scale(scale_co);

        for(int i=0;i<Nbin;i++)
        {
        N2[i] = N2[i]+k;
	NBr2[i] = NBr2[i]+k;
        }

	// for(int i=0;i<numb;i++)
	// {
	// hraw2->SetBinContent(i+1,hraw2->GetBinContent(i+1)+k);
	// }

        // hraw->SetTitle("");
        // hraw->GetXaxis()->SetTitle("Energy (keVee)");
        // hraw->GetXaxis()->SetTitleFont(22);
        // hraw->GetXaxis()->SetTitleSize(0.05);
        // hraw->GetXaxis()->SetTitleOffset(0.9);
        // hraw->GetXaxis()->SetLabelSize(0.05);
        // hraw->GetXaxis()->SetLabelFont(22);
        // //hraw->GetXaxis()->CenterTitle(kTRUE);
        // hraw->GetXaxis()->SetRangeUser(0,12);
        // hraw->GetYaxis()->SetTitle("Counts");
        // hraw->GetYaxis()->SetTitleSize(0.05);
        // hraw->GetYaxis()->SetTitleFont(22);
        // hraw->GetYaxis()->SetTitleOffset(0.9);
        // hraw->GetYaxis()->SetLabelSize(0.05);
        // hraw->GetYaxis()->SetLabelFont(22);
        // //hraw->GetYaxis()->CenterTitle(kTRUE);
        // hraw->GetYaxis()->SetRangeUser(0,3e2);
        // //hraw->GetYaxis()->SetLabelOffset(0.002);
        // hraw->SetLineColor(2);
        // hraw->SetLineWidth(3);
        // hraw->SetLineStyle(1);
        // hraw->Draw();

        // hraw2->SetLineColor(6);
        // hraw2->SetLineWidth(3);
        // hraw2->SetLineStyle(1);
        // hraw2->Draw("same");

TMultiGraph *mg = new TMultiGraph();
TGraphAsymmErrors *g_N1 = new TGraphAsymmErrors(Nbin,energy,N1,ee,ee,eN1_d,eN1_u);
g_N1->SetMarkerStyle(20);
g_N1->SetMarkerColor(1);
g_N1->SetLineColor(1);
g_N1->SetLineWidth(2);
//mg->Add(g_N1);

TGraphAsymmErrors *g_NBr1 = new TGraphAsymmErrors(Nbin,energy,NBr1,ee,ee,edNBr1,euNBr1);
g_NBr1->SetMarkerStyle(21);
g_NBr1->SetMarkerColor(1);
g_NBr1->SetLineColor(1);
g_NBr1->SetLineWidth(1);
mg->Add(g_NBr1);

TGraphAsymmErrors *g_N2 = new TGraphAsymmErrors(Nbin,energy,N2,ee,ee,eN2_d,eN2_u);
g_N2->SetMarkerStyle(20);
g_N2->SetMarkerColor(4);
g_N2->SetLineColor(4);
g_N2->SetLineWidth(2);
//mg->Add(g_N2);

TGraphAsymmErrors *g_NBr2 = new TGraphAsymmErrors(Nbin,energy,NBr2,ee,ee,edNBr2,euNBr2);
g_NBr2->SetMarkerStyle(21);
g_NBr2->SetMarkerColor(4);
g_NBr2->SetLineColor(4);
g_NBr2->SetLineWidth(1);
mg->Add(g_NBr2);

mg->Draw("AP");
/*
mg->GetXaxis()->SetTitle("E (keVee)");
mg->GetXaxis()->SetTitleFont(62);
mg->GetXaxis()->SetTitleSize(0.05);
mg->GetXaxis()->SetTitleOffset(0.9);
mg->GetXaxis()->SetLabelSize(0.05);
mg->GetXaxis()->SetLabelFont(62);
mg->GetXaxis()->CenterTitle(kTRUE);
mg->GetXaxis()->SetLimits(0.15,11.5);
mg->GetYaxis()->SetTitle("counts (kg^{-1}keVee^{-1}day^{-1})");
mg->GetYaxis()->SetTitleSize(0.05);
mg->GetYaxis()->SetTitleFont(62);
mg->GetYaxis()->SetTitleOffset(0.9);
mg->GetYaxis()->SetLabelSize(0.05);
mg->GetYaxis()->SetLabelFont(62);
mg->GetYaxis()->CenterTitle(kTRUE);
mg->GetYaxis()->SetLabelOffset(0.002);
*/
//mg->SetMinimum(1);
//mg->SetMaximum(1e7);
c1->Update();

TLegend *legend = new TLegend(0.6,0.63,0.88,0.88);
legend->SetLineColor(0);
legend->SetFillColor(0);
legend->SetFillStyle(1001);
//legend->AddEntry(g_N1,"^{137}Cs #otimes Bm","lep");
legend->AddEntry(g_NBr1,"^{137}Cs #otimes B_{r}","lep");
// legend->AddEntry(hraw,"^{137}Cs #otimes Simulation","l");
//legend->AddEntry(g_N2,"^{60}Co #otimes Bm + 150","lep");
legend->AddEntry(g_NBr2,"^{60}Co #otimes B_{r}","lep");
// legend->AddEntry(hraw2,"^{60}Co #otimes Simulation","l");
legend->SetTextFont(22);
legend->SetTextSize(0.04);
legend->Draw();
c1->Update();

	// TFile *outfile = new TFile("C1B_spectrum_source_br_201708_latest.root","recreate");
	// c1->Write("c1",TObject::kOverwrite);
   //      hraw->Write("hraw",TObject::kOverwrite);
   //      hraw2->Write("hraw2",TObject::kOverwrite);
   //      outfile->Close();

}

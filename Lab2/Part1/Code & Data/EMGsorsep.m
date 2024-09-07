function [sources,A_x,autocor]= EMGsorsep(sig)
disp('CCA')
%[sources,w_x,autocor]= EMGsorsep(sig)
%
%  input  : [sig], EEG matrix dimension(channels x time samples);
%  output : [sources], time course of the sources obtained by BSS-CCA;
%         : [w_x], demixing matrix;
%         : [autocor], autocorrelation of the sources;
%
%© 2005 Wim De Clercq 
%Notice that the BSS-CCA software (containing EMGscrol.m and EMGsorsep.m) can be freely used for non-commercial use only. For commercial licenses, see Commercial Use. 
%The BSS-CCA software is accompanied by an Academic Public License. 
%These licenses can be found at:
%[Engels] http://www.neurology-kuleuven.be/?id=210
%[Nedederlands] http://www.neurology-kuleuven.be/?id=209
%If utilization of the BSS-CCA software results in outcomes which will be published,  
%Academic User shall acknowledge K.U.LEUVEN as the provider of the BSS-CCA software and shall 
%include a reference to [Vergult A., De Clercq W., Palmini A., Vanrumste B., Dupont P., Van Huffel S., Van Paesschen W., ``Improving the Interpretation of Ictal Scalp EEG: BSS-CCA algorithm for muscle artifact removal'', Internal Report 06-148, ESAT-SISTA, K.U.Leuven (Leuven, Belgium), 2006.] 
%and [De Clercq W., Vergult A., Vanrumste B., Van Paesschen W., Van Huffel S., ``Canonical Correlation analysis applied to remove muscle artifacts from the electroencephalogram'', Internal Report 05-116, ESAT-SISTA, K.U.Leuven (Leuven, Belgium), 2005. Accepted for publication in Transactions on Biomedical Engineering. ] in the manuscript. 

N=size(sig);
for i=1:N(1)
sig(i,:)=sig(i,:)-mean(sig(i,:));
end
x=sig(:,1:end-1);
y=sig(:,2:end);
[Q_x,R_x]=qr(x',0);
[Q_y,R_y]=qr(y',0);
[U,S,V]=svd(Q_x'*Q_y);
sources=(Q_x*U)';
autocor=diag(S);
w_x=(inv(R_x)*U);
A_x=pinv(w_x)';


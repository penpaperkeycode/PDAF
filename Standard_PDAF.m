function [TrackList]=Standard_PDAF(TrackList)

for i=1:size(TrackList,2) %Need any track, witghout it, it won't run
    if TrackList{i}.perish ~= 1
        
        % extract track coordinates and covariance
        x = TrackList{i}.x;
        P = TrackList{i}.P;
        F = TrackList{i}.F;
        H = TrackList{i}.H;
        Q = TrackList{i}.Q;
        R = TrackList{i}.R;
        
        % make prediction
        xpred   = F*x;
        Ppred   = F*P*F' + Q;
        zpred   = H*xpred;
        S_tmp       = H*Ppred*H' + R;
        S = (S_tmp + S_tmp.') / 2;  %% For Positive definite / Symmetric / Square
        
        %--------------Process the measurements and do Probabilistic Data Association
        if isempty(TrackList{i}.Measurement)
            z=zpred; %No measurement case
            v=zeros(size(z,1),1);
            
            %For Probability Data Association
            Beta_0=1; %switching value
            
            %Calc Kalman gain
            K=(Ppred*H')/(H*Ppred*H'+R);
            P_c=Ppred-K*S*K';
            P_tilda=0;
            
        else
            z=TrackList{i}.Measurement;
            %For Probability Data Association
            Beta=[]; LR=[]; Sum_LR=0; v=zeros(size(z,1),1); Beta_0=0; %switching value
            
            %Calc LR
            for j=1:size(TrackList{i}.Measurement,2)
                LR(j)=mvnpdf(z(:,j),zpred,S);
                %The error may occur on 'mvnpdf' : SIGMA must be a square, symmetric, positive definite matrix.
                Sum_LR=Sum_LR+LR(j);
            end
            
            %Calc Beta
            for j=1:size(TrackList{i}.Measurement,2)
                Beta(j)=LR(j)/Sum_LR;
            end
            
            %Calc Combined Innovation
            for j=1:size(TrackList{i}.Measurement,2)
                v=v+Beta(j)*(z(:,j)-zpred);
            end
            
            %Calc Kalman gain
            K=(Ppred*H')/(H*Ppred*H'+R);
            P_c=Ppred-K*S*K';
            
            %----------------for Ptilda------------
            Spread_of_Innovations_temp=0;
            for j=1:size(TrackList{i}.Measurement,2)
                v_l=(z(:,j)-zpred);
                Spread_of_Innovations_temp=Spread_of_Innovations_temp+(Beta(j)*(v_l*v_l'));
            end
            
            P_tilda=K*(Spread_of_Innovations_temp-v*v')*K';
            
        end
        
        x=xpred+K*v;
        %----------------------------------------------
        P=Beta_0*Ppred+(1-Beta_0)*P_c+P_tilda;
        %----------------------------------------------
        TrackList{i}.x=x;   %Parameter Update
        TrackList{i}.P=P;
        
    end
    
end
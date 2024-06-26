function X_estimate=PoissonMBMtarget_estimate2(filter_upd)

%Author: Angel F. Garcia Fernandez 


%We do as in the delta-GLMB: First MAP of cardinality, them MAP of
%cardinality with highest weight

if(~isempty(filter_upd.globHyp))
    
    globHyp=filter_upd.globHyp;
    globHypWeight=filter_upd.globHypWeight;
    Nhyp=length(globHypWeight);
    N_tracks=length(filter_upd.tracks);
    
    pcard_tot=zeros(1,N_tracks+1); %Cardinality distribution of the PMBM density
    eB_tot=zeros(Nhyp,N_tracks);
    for j=1:Nhyp
        eB_hyp_j=zeros(1,N_tracks);
        for i=1:N_tracks
            hyp=globHyp(j,i);
            if(hyp~=0)
                eB=filter_upd.tracks{i}.eB(hyp);
                eB_hyp_j(i)=eB;
                
            end
        end
        eB_tot(j,:)=eB_hyp_j;
        pcard_j=CardinalityMB(eB_hyp_j);
        %Multiply by weights as well
        pcard_tot=pcard_tot+globHypWeight(j)*pcard_j;
    end
    
    %We estimate the cardinality
    [m,index]=max(pcard_tot);
    card_estimate=index-1;
    
    if(card_estimate>0)
        %Now we go through all hypotheses again
        weight_hyp_card=zeros(1,Nhyp);
        indices_sort_hyp=zeros(Nhyp,N_tracks);
        
        for j=1:Nhyp
            eB_hyp_j=eB_tot(j,:);
            [eB_hyp_j_sort,indices_sort]=sort(eB_hyp_j,'descend');            
            indices_sort_hyp(j,:)=indices_sort;
            %We compute the product (and also account for global hypothesis weight)
            vector=[eB_hyp_j_sort(1:card_estimate),1-eB_hyp_j_sort(card_estimate+1:end)];
            weight_hyp_card(j)=globHypWeight(j)*prod(vector);
                      
        end
        
        [m_f,index_f]=max(weight_hyp_card);
        indices_sort=indices_sort_hyp(index_f,:);
        hyp=globHyp(index_f,:);
        X_estimate=zeros(4*card_estimate,1);
        
        for i=1:card_estimate
            target_i=indices_sort(i);
            hyp_i=hyp(target_i);
            X_estimate(4*i-3:4*i)=filter_upd.tracks{target_i}.meanB{hyp_i};    
        end
        
        
        
        
        
    else
        
        X_estimate=[];
    end
    
else
    X_estimate=[];
end
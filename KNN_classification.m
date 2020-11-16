
close all;
clear all;

ratio = input('What is the ratio of TrainData: '); %e�itim oran�
knn = input('number of K ?: ');                    %k say�s�

HSI = load('salinas.mat');            %hiperspektral g�r�nt�
HSI = HSI.salinas;
HSI_GT = load('salinas_gt.mat');      %Groundtruth
HSI_GT = HSI_GT.salinas_gt;

[spat1,spat2,spec]=size(HSI);
[row, col, val] = find(HSI_GT);       %val= toplam etikektli piksel say�s�
Num_of_Classes = max(HSI_GT(:));      %ka� tane s�n�f oldu�u bulunur

t=0;
for n=1:1:Num_of_Classes   %s�n�f say�s� kadar d�g�ye girsin
    [row, col, val] = find(HSI_GT == n); %bir s�n�f�n sat�r s�t�n ve etiket bilgilerini bul
    [Num_of_Train , one] = size(val);    
    a=round(Num_of_Train*ratio);
    b=randperm(a,a);            % rasgele pikseller se�me
    for i=1:1:a
        label_location(i+t,1)=row(b(i));     %e�itim i�in al�nan piksellerin sat�r sut�n ve etiket bilgisini ayr� bir matriste tut
        label_location(i+t,2)=col(b(i));
        label_location(i+t,3)=HSI_GT(row(b(i)),col(b(i)));
    end
    t=t+a;
end

% K-NN s�nfland�rma
distance=0;
counter=size(label_location,1);
neighbors=zeros(1,knn);
tagged=zeros(spat1,spat2); %s�n�fland�rma sonucunu g�sterecek matris
  for x=1:spat1 
      for y=1:spat2   
          for z=1:counter %eldeki b�t�n e�itim �rneklerini dola�acak
            for band=1:spec
                distance = distance + (HSI(x,y,band)-HSI(label_location(z,1),label_location(z,2),band))^2; %�klid mesafesi
            end
            dist(z,1)=sqrt(distance);
            distance=0;
          end
          [v , index]=sort(dist(:,1)); %en d���k mesafeye sahip olana g�re s�rala
          for k=1:knn
            neighbors(k)=label_location(index(k),3); %K say�s� kadar en k�sa mesafeye sahip �rnek etiket de�erini al
          end
          tagged(x,y)=mode(neighbors); % en �ok tekrarlayan� se�                   
      end
  end
  
 figure;
 subplot(1,2,1); imagesc(tagged); title('KNN Classified Hyperspectral Image');
 subplot(1,2,2); imagesc(HSI_GT); title('Groundtruth');
 
 % ba�ar�m oran�
 true=0;
 false=0;
 for x=1:spat1 
      for y=1:spat2
          if(HSI_GT(x,y)~=0)
              if(tagged(x,y)==HSI_GT(x,y))
                  true = true+1;
              else
                  false = false+1;
              end
          end
      end
 end
 success_rate= true*100 / (true + false);
 print=['overall accuracy of KNN Classification = %',num2str(success_rate)];
 disp(print)
  



 
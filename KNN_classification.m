
close all;
clear all;

ratio = input('What is the ratio of TrainData: '); %eðitim oraný
knn = input('number of K ?: ');                    %k sayýsý

HSI = load('salinas.mat');            %hiperspektral görüntü
HSI = HSI.salinas;
HSI_GT = load('salinas_gt.mat');      %Groundtruth
HSI_GT = HSI_GT.salinas_gt;

[spat1,spat2,spec]=size(HSI);
[row, col, val] = find(HSI_GT);       %val= toplam etikektli piksel sayýsý
Num_of_Classes = max(HSI_GT(:));      %kaç tane sýnýf olduðu bulunur

t=0;
for n=1:1:Num_of_Classes   %sýnýf sayýsý kadar dögüye girsin
    [row, col, val] = find(HSI_GT == n); %bir sýnýfýn satýr sütün ve etiket bilgilerini bul
    [Num_of_Train , one] = size(val);    
    a=round(Num_of_Train*ratio);
    b=randperm(a,a);            % rasgele pikseller seçme
    for i=1:1:a
        label_location(i+t,1)=row(b(i));     %eðitim için alýnan piksellerin satýr sutün ve etiket bilgisini ayrý bir matriste tut
        label_location(i+t,2)=col(b(i));
        label_location(i+t,3)=HSI_GT(row(b(i)),col(b(i)));
    end
    t=t+a;
end

% K-NN sýnflandýrma
distance=0;
counter=size(label_location,1);
neighbors=zeros(1,knn);
tagged=zeros(spat1,spat2); %sýnýflandýrma sonucunu gösterecek matris
  for x=1:spat1 
      for y=1:spat2   
          for z=1:counter %eldeki bütün eðitim örneklerini dolaþacak
            for band=1:spec
                distance = distance + (HSI(x,y,band)-HSI(label_location(z,1),label_location(z,2),band))^2; %öklid mesafesi
            end
            dist(z,1)=sqrt(distance);
            distance=0;
          end
          [v , index]=sort(dist(:,1)); %en düþük mesafeye sahip olana göre sýrala
          for k=1:knn
            neighbors(k)=label_location(index(k),3); %K sayýsý kadar en kýsa mesafeye sahip örnek etiket deðerini al
          end
          tagged(x,y)=mode(neighbors); % en çok tekrarlayaný seç                   
      end
  end
  
 figure;
 subplot(1,2,1); imagesc(tagged); title('KNN Classified Hyperspectral Image');
 subplot(1,2,2); imagesc(HSI_GT); title('Groundtruth');
 
 % baþarým oraný
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
  



 
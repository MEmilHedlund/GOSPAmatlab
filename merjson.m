function [z_t, intensity_clutter, ntargets, postruths, X_truth] = getmeas(beginning,blocklength,json_file)

    fid = fopen(json_file, 'r');
    if fid == -1
        disp(['Failed to open JSON file: ', json_file]);
        return;
    end

    all_data = {};
    line_count = 0;
    while ~feof(fid)
        line = fgetl(fid);
        line_count = line_count + 1;
        try
            json_obj = jsondecode(line);
            all_data{line_count} = json_obj;
        catch ME
            disp(['Error decoding JSON at line ', num2str(line_count), ': ', ME.message]);
            disp(['Failed JSON string: ', line]);
        end
    end
    fclose(fid);
    area=100^2*pi*(70/360);
    k=0;
    X_truth=[0];
    
    for i = beginning:beginning+blocklength-1
        k=k+1;
        data=all_data{i};
        numtargets=0;
        clutter=0;
        poslen=0;
        
        for l = 1:length(data{1}.ids_truth)
            car=fieldnames(data{1}.bboxes_truth);
            X_truth(poslen*4+1,k)=data{1}.bboxes_truth.(car{l}){1}(1);
            X_truth(poslen*4+2,k)=data{1}.bboxes_truth.(car{l}){2}(1);
            X_truth(poslen*4+3,k)=data{1}.bboxes_truth.(car{l}){1}(5);
            X_truth(poslen*4+4,k)=data{1}.bboxes_truth.(car{l}){2}(2);
            postruth(1,poslen+1)=sum(data{1}.bboxes_truth.(car{l}){1}(1:4))/4;
            postruth(2,poslen+1)=sum(data{1}.bboxes_truth.(car{l}){1}(5:8))/4;
            poslen=poslen+1;
        end
        
        for j = 2:length(data)
            z(1,j-1)=data{j}.pointcloudx;
            z(2,j-1)=data{j}.pointcloudy;
            if data{j}.tag~=14
                clutter=clutter+1;
            end
        end
        intensity_clutter(k)=clutter/area;
        ntargets(k)=length(data{1}.ids_truth);
        z_t{k}=z;
        postruths{k}=postruth;
        clear z
    end
    intensity_clutter=sum(intensity_clutter)/length(intensity_clutter);
end


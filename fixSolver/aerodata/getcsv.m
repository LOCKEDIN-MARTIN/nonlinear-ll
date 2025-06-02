aerodata = load("aerodata.mat");

dim = size(aerodata.aerodata);
for i = 1 : dim(end)

    name = strcat('aerodata_aileron',string(i),'.csv');
    csvwrite(name,aerodata.aerodata(:, :, i));

end
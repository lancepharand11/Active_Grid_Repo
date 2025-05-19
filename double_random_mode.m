%% User input
clear;
clc;

% Inputs from model
Grid_Re = 40000;
rossbyNumber = 21;
stdDev_Velo = 0.000696; %std deviation of angular velocity

%% Motor parameter calcs
N = 20; %number of motors
c = 0.1; % m chord length
kinematicVisc_Air = 1.516 * 10^(-5); %m^2/s, Used air at 20 C

% % Using Rossby Number to modify integral length scale size (Ro is 
% % proportional to Lxx)
% % Sets the rps of the shaft
meshLength = 60.96 * 10^(-3); %m
freestream_Velo = Grid_Re * kinematicVisc_Air / meshLength; %m/s, mean freestream velocity
rpsVelo = freestream_Velo / (rossbyNumber * meshLength); %rev/s

%Setup mean and std dev cruise time. Used same cruise time as UofT paper
mean_ConvectiveTime = 100;
mean_Time = mean_ConvectiveTime/freestream_Velo*c; %s, mean cruise time
stdDev_Time = mean_Time/(3/2*sqrt(3)); %std deviation of cruise time

%Setup mean and std dev angular velo
mean_Velo = rpsVelo; %rev/s, mean angular velocity

%% Initate System 
port = 7775;

%u = 1; %COMMENT THIS OUT. ONLY FOR DEBUGGING
u = udpport("LocalHost","192.168.1.10","LocalPort",1000);
configureTerminator(u,"CR");

for i = 1:N
    %initialize shafts with 5 rps
    first_command = 'JS5';
    %How does this work?
    first_bytesToSend = [0,7,uint8(first_command),13];
    
    %stop jogging - Why is this needed?
    second_command = 'CJ';
    %How does this work?
    second_bytesToSend = [0,7,uint8(second_command),13];

    bytesToSend = [first_bytesToSend,second_bytesToSend];
    write(u, bytesToSend, "192.168.1." + string((i + 10)), port);
end


%% Main script 

%motor number vector
motorNum = (1:N).';

while(true) 
    %execute random generation function
    [r_Velo, r_Time] = rps_cruiseTime_Gen(N, mean_Time, stdDev_Time, mean_Velo, stdDev_Velo);

    %store velocity, time, and motor number in a matrix
    dataArray = [r_Time r_Velo motorNum];
    
    %sort array in order of increasing cruise time
    sortedData = sortrows(dataArray, {'ascend'});
    
    disp("start") %FOR DEBUGGING

    %start loop timer
    startLoop = tic;
    
    %store highest cruise time in variable
    endTime = sortedData(N, 1);
    
    %store an unmodified copy of sortedData array
    origSortedData = sortedData;

    %condition: timer less than highest generated cruise time
    while(toc(startLoop) <= endTime)
        for i = 1:N
            disp("next motor") %FOR DEBUGGING
            disp(origSortedData(i, 1)) %FOR DEBUGGING
            pause(sortedData(i, 1)) %pause by current cruise time amount
            
            %remove current cruise time from other cruise times
            sortedData(i:N, 1) = sortedData(i:N, 1) - sortedData(i, 1);
            
            %start timer for in loop
            inLoopStart = tic;

            %change jog speed for motor i
            first_command = "CS" + sortedData(i, 2);
            %How does this work?
            first_bytesToSend = [0,7,uint8(char(first_command)),13];
            
            % %stop jogging - Why is this needed?
            % second_command = 'SJ';
            % %How does this work?
            % second_bytesToSend = [0,7,uint8(second_command),13];
            
            %store bytes in 1 matrix
            bytesToSend = first_bytesToSend;
            
            % send bytes
            write(u, bytesToSend, "192.168.1." + string(sortedData(i, 3) + 10), port);
            
            %store time taken to ran section of loop
            inLoopEnd = toc(inLoopStart);

            if i == N
                continue
            else
                %remove time taken to execute section of loop from all
                %cruise times
                sortedData((i + 1):N, 1) = sortedData((i + 1):N, 1) - inLoopEnd;
            end

        end

    end

end



% EXTRA:
% write(u,bytesToSend,"192.168.1.12",port);
% write(u,bytesToSend,"192.168.1.13",port);
% write(u,bytesToSend,"192.168.1.14",port);
% write(u,bytesToSend,"192.168.1.15",port);
% write(u,bytesToSend,"192.168.1.16",port);
% write(u,bytesToSend,"192.168.1.17",port);
% write(u,bytesToSend,"192.168.1.18",port);
% write(u,bytesToSend,"192.168.1.19",port);
% write(u,bytesToSend,"192.168.1.20",port);
% write(u,bytesToSend,"192.168.1.21",port);
% write(u,bytesToSend,"192.168.1.22",port);
% write(u,bytesToSend,"192.168.1.23",port);
% write(u,bytesToSend,"192.168.1.24",port);
% write(u,bytesToSend,"192.168.1.25",port);
% write(u,bytesToSend,"192.168.1.26",port);
% write(u,bytesToSend,"192.168.1.27",port);
% write(u,bytesToSend,"192.168.1.28",port);
% write(u,bytesToSend,"192.168.1.29",port);
% write(u,bytesToSend,"192.168.1.30",port);
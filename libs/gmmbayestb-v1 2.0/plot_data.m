function plot_data(data, type, colors)

for k = 1:max(type)
	x = data(type==k,1);
	y = data(type==k,2);
	if ~isempty(x)
		h = plot(x, y, colors(mod(k-1,size(colors,1))+1,:));
	end
	%set(h, 'MarkerSize', msize);
	hold on
end
hold off
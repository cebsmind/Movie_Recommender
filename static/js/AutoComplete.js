$(function () {
    $("#movieTitleInput").autocomplete({
        source: function (request, response) {
            // Fetch matching movie titles from backend
            $.ajax({
                url: '/autocomplete',
                data: { partial_title: request.term },
                success: function (data) {
                    response(data);
                }
            });
        },
        minLength: 2
    });
});
<%@ Page Title="DataGetter Page" Language="C#" MasterPageFile="~/Site.Master" AutoEventWireup="true" CodeBehind="Default.aspx.cs" Inherits="DataGetter._Default" %>
<asp:Content ID="BodyContent" ContentPlaceHolderID="MainContent" runat="server">
    <section>
        <asp:Label ID="LabelSessionID" runat="server" Text="SessionID"></asp:Label>
        <asp:TextBox ID="TextBoxSessionID" runat="server"></asp:TextBox>
    </section>
    <section>
        <asp:Label ID="LabelSubSessionID" runat="server" Text="SubSessionID"></asp:Label>
        <asp:TextBox ID="TextBoxSubSessionID" runat="server"></asp:TextBox>
    </section>
    <section>
         <asp:Button ID="Button1" runat="server" Text="GetData" OnClick="CommandGetData" />
    </section>
</asp:Content>
